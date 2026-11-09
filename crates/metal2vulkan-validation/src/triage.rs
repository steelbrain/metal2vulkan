use crate::check::cfg_fact::has_cfg_cycle;
use crate::hash::sha256_bytes;
use crate::requirement::{Executor, Limit, ToolingRequirement, UnobservedEffect};
use crate::source::SourceRow;
use rusqlite::{params, Connection, OptionalExtension, Transaction};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// The classifier's contract version: a name for the classification semantics that the
/// fingerprints below cannot see (what a cached fact's fields mean). Edits to the classifier's
/// sources change the key by themselves; bump this when a fact's meaning changes. The commit that
/// bumps it says what changed; this comment keeps no history.
macro_rules! analyzer_version {
    () => {
        "structural-triage-v81"
    };
}

/// The cache key of every persisted [`StructuralTriage`].
///
/// Cache validity follows semantics (AGENTS.md): the key covers exactly what classification reads.
/// That is the product it calls (`inspect` reflection, the intrinsic and command-encoder
/// contracts) and this crate's classifier sources, each fingerprinted by the build: the product's
/// sources, and the validation files named by `METAL2VULKAN_CLASSIFIER_INPUTS`
/// (`validation/build.rs`). An edit to either invalidates every cached fact with no hand bump to
/// forget; an edit to any other validation code leaves them current.
pub const ANALYZER_ABI: &str = concat!(
    analyzer_version!(),
    ":",
    env!("METAL2VULKAN_PRODUCT_FINGERPRINT"),
    ":",
    env!("METAL2VULKAN_CLASSIFIER_FINGERPRINT")
);

const RESOURCE_KINDS: [&str; 13] = [
    "air.buffer",
    "air.texture",
    "air.sampler",
    "air.imageblock",
    "air.visible_function_table",
    "air.intersection_function_table",
    "air.acceleration_structure",
    "air.command_buffer",
    "air.threadgroup_position_in_grid",
    "air.thread_position_in_grid",
    "air.thread_position_in_threadgroup",
    "air.vertex_input",
    "air.render_target",
];

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralTriage {
    pub signature: String,
    pub has_cfg_cycle: bool,
    pub air_calls: BTreeMap<String, usize>,
    pub unrecognized_air_intrinsics: BTreeMap<String, usize>,
    pub resource_markers: BTreeMap<String, usize>,
    pub uses_device_addresses: bool,
    pub uses_indirect_commands: bool,
    /// The entry is a fragment function with an explicit imageblock (the reflection's
    /// `fragment_imageblock`: `air.imageblock_data` views of a master layout). Descriptive, like
    /// `uses_indirect_commands`: every executor runs the family.
    #[serde(default)]
    pub fragment_imageblock: bool,
    /// The entry's argument buffers hold a compute or render pipeline-state member (the
    /// reflection's `argument_buffer_fields`). Descriptive, like `uses_indirect_commands`: every
    /// executor runs the family.
    #[serde(default)]
    pub pipeline_state_members: bool,
    /// The entry is a tile function: a kernel with implicit imageblock attachments (the
    /// reflection's `implicit_imageblock_attachments`, which the checker pairs with render targets
    /// and `imageblock.implicit_coverage`). Descriptive, like `fragment_imageblock`: every executor
    /// runs the family, and the focused audit selects it by this fact
    /// ([`AuditTarget::TileFunctions`]) whatever its requirements.
    #[serde(default)]
    pub tile_function: bool,
    /// An object or mesh entry's draw: its stage, its payload, and the stages a row of it needs
    /// that nothing supplies ([`crate::executor_contract::mesh::MeshDrawFacts`]). Descriptive, like
    /// `tile_function`: the focused audit selects the family by this fact
    /// ([`AuditTarget::MeshDraws`]) whatever its requirements.
    #[serde(default)]
    pub mesh_draw: Option<crate::executor_contract::mesh::MeshDrawFacts>,
    /// The module maps coordinates through a rasterization-rate map (an
    /// `air.map_screen_to_physical_coordinates` or `air.map_physical_to_screen_coordinates` call,
    /// any overload). Descriptive, like `tile_function`: where the calls read their maps decides
    /// `rasterization_rate_map_placement`, and the focused audit selects the family by this fact
    /// ([`AuditTarget::RasterizationRateMaps`]) whatever its requirements.
    #[serde(default)]
    pub rasterization_rate_map: bool,
    /// The sample-rate features the entry uses
    /// ([`crate::executor_contract::sample_rate::sample_rate_uses`]: the raster sample count, the
    /// sample index and coverage inputs, a coverage output, a varying sampled per sample).
    /// Descriptive, like `rasterization_rate_map`: both executors run a multisampled pass, and the
    /// focused audit selects the family by this fact ([`AuditTarget::SampleRate`]) whatever its
    /// requirements.
    #[serde(default)]
    pub sample_rate: Vec<crate::executor_contract::sample_rate::SampleRateUse>,
    /// The module calls a texture operation on a Metal `ulong` texture
    /// ([`crate::executor_contract::calls_ulong_texture`]). Descriptive, like `sample_rate`:
    /// both executors run the family, and the focused audit selects it by this fact
    /// ([`AuditTarget::UlongTextures`]) whatever its requirements.
    #[serde(default)]
    pub ulong_texture: bool,
    /// The declared type of each varying the entry reads, in location order, when it is a
    /// fragment function (the reflection's `varyings`; `?` where the AIR names no type); empty
    /// for any other entry. Descriptive, like `ulong_texture`: both executors draw a fragment case
    /// through the generated passthrough vertex, which writes each varying its default value or
    /// the case's authored one (`AuthoredCase::varying_values`), and the focused audit selects the
    /// family by this fact ([`AuditTarget::FragmentVaryings`]) whatever its requirements.
    #[serde(default)]
    pub fragment_varyings: Vec<String>,
    /// The logical symbols of the visible functions the entry references and links from another
    /// module (its projection's `air.visible_function_reference` metadata, sorted, once each; an
    /// unreadable reference as `<the reason>`). Descriptive, like `fragment_varyings`: a case links
    /// each from a retained module or a public module compiled from owned MSL, which both executors
    /// build, and the focused audit selects the family by this fact
    /// ([`AuditTarget::LinkedFunctions`]) whatever its requirements.
    #[serde(default)]
    pub linked_functions: Vec<String>,
    /// Calls whose effect both executors run exactly but no observation captures, by effect.
    /// Descriptive, like `uses_indirect_commands`: kept apart from `tooling_requirements`, and
    /// left out of symbol coverage so no report counts these calls as verified.
    pub unobserved_calls: BTreeMap<String, UnobservedEffect>,
    pub tooling_requirements: Vec<ToolingRequirement>,
}

impl StructuralTriage {
    /// The distinct effects this row's cases cannot observe, for per-row reports.
    pub fn unobserved_effects(&self) -> BTreeSet<UnobservedEffect> {
        self.unobserved_calls.values().copied().collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuditTarget {
    IndirectCommands,
    /// Modules that map coordinates through a rasterization-rate map
    /// ([`StructuralTriage::rasterization_rate_map`]).
    RasterizationRateMaps,
    /// Entries that use a sample-rate feature ([`StructuralTriage::sample_rate`]).
    SampleRate,
    /// Modules that operate on a Metal `ulong` texture ([`StructuralTriage::ulong_texture`]).
    UlongTextures,
    /// Fragment functions that read varyings ([`StructuralTriage::fragment_varyings`]).
    FragmentVaryings,
    /// Entries that reference visible functions of another module
    /// ([`StructuralTriage::linked_functions`]).
    LinkedFunctions,
    /// Tile functions: kernels with implicit imageblock attachments
    /// ([`StructuralTriage::tile_function`]).
    TileFunctions,
    /// Object and mesh functions ([`StructuralTriage::mesh_draw`]).
    MeshDraws,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AuthoringCapabilitySummary {
    pub total_sources: usize,
    pub classified_sources: usize,
    /// Classified sources the executors cannot evidence: a tooling requirement
    /// ([`ToolingRequirement::blocks_closure`]), or an `air.*` call the engine's intrinsic
    /// table does not know.
    pub unresolved_sources: usize,
    /// Classified sources Metal itself cannot build a pipeline for (a requirement whose limit is
    /// [`Limit::Metal`]): closed without a Metal reference, and unresolved only when a tooling gap
    /// of their own keeps them open.
    pub metal_refused_sources: usize,
    pub requirements: BTreeMap<ToolingRequirement, usize>,
    /// The classified sources each executor cannot run, and whose limit stops it (an executor that
    /// no requirement stops is absent).
    pub executor_stops: BTreeMap<Executor, ExecutorStops>,
    /// Classified sources with at least one call of each unobserved effect. Not unresolved: the
    /// sources are authorable, but their cases do not evidence these effects.
    pub unobserved_effects: BTreeMap<UnobservedEffect, usize>,
    pub unrecognized_air_intrinsics: BTreeMap<String, usize>,
}

/// The classified sources one executor cannot run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ExecutorStops {
    /// Sources with at least one requirement that stops the executor.
    pub sources: usize,
    /// Those with a tooling requirement ([`Limit::Tooling`]) that stops it: each keeps the census
    /// open.
    pub tooling_gap: usize,
    /// Those with a requirement whose limit is Metal's own ([`Limit::Metal`]): closed without a
    /// reference, unless a tooling gap of their own keeps them open.
    pub metal_refused: usize,
}

impl AuthoringCapabilitySummary {
    pub fn remaining_sources(&self) -> usize {
        self.total_sources.saturating_sub(self.classified_sources)
    }

    /// Why a complete census is not closed: its unresolved rows. Every entry of every module is a
    /// row, so no module is outside the census. `None` while rows remain unclassified (a bounded
    /// batch reports no verdict) or when the census is closed.
    pub fn closure_gap(&self) -> Option<String> {
        if self.remaining_sources() != 0 || self.unresolved_sources == 0 {
            return None;
        }
        Some(format!(
            "{} source(s) still have validation support gaps",
            self.unresolved_sources
        ))
    }
}

/// Select sources from the versioned triage cache by an exact tooling requirement.
///
/// This query touches only SQLite. The caller can then use indexed byte locations to read the
/// bounded set of source rows it intends to audit.
pub fn select_cached_requirement(
    index: &Path,
    requirement: ToolingRequirement,
    limit: usize,
) -> Result<Vec<String>, String> {
    select_cached_requirement_after(index, requirement, None, limit)
}

pub fn select_cached_requirement_after(
    index: &Path,
    requirement: ToolingRequirement,
    after: Option<&str>,
    limit: usize,
) -> Result<Vec<String>, String> {
    let connection = Connection::open(index)
        .map_err(|error| format!("open index {}: {error}", index.display()))?;
    ensure_cache_table(&connection)?;
    let mut statement = connection
        .prepare(
            "SELECT s.air_sha256 FROM sources s JOIN triage_analysis t USING (air_sha256) \
             WHERE t.analyzer_abi=?1 AND (?2 IS NULL OR s.air_sha256>?2) AND EXISTS (\
               SELECT 1 FROM json_each(t.result_json, '$.tooling_requirements') r \
               WHERE r.value=?3\
             ) ORDER BY s.air_sha256 LIMIT ?4",
        )
        .map_err(|error| format!("prepare cached requirement query: {error}"))?;
    let hashes = statement
        .query_map(
            params![ANALYZER_ABI, after, requirement.as_str(), limit as i64],
            |row| row.get::<_, String>(0),
        )
        .map_err(|error| format!("query cached requirement {requirement}: {error}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("read cached requirement {requirement}: {error}"))?;
    Ok(hashes)
}

/// Select a supported structural family for focused regression auditing.
///
/// Audit targets are deliberately independent of [`ToolingRequirement`]: gaining support must not
/// make the corresponding regression audit select zero rows.
pub fn select_cached_audit_target_after(
    index: &Path,
    target: AuditTarget,
    after: Option<&str>,
    limit: usize,
) -> Result<Vec<String>, String> {
    let connection = Connection::open(index)
        .map_err(|error| format!("open index {}: {error}", index.display()))?;
    ensure_cache_table(&connection)?;
    let predicate = match target {
        AuditTarget::IndirectCommands => {
            "json_extract(t.result_json, '$.uses_indirect_commands') = 1"
        }
        AuditTarget::RasterizationRateMaps => {
            "json_extract(t.result_json, '$.rasterization_rate_map') = 1"
        }
        AuditTarget::SampleRate => {
            "coalesce(json_array_length(t.result_json, '$.sample_rate'), 0) > 0"
        }
        AuditTarget::UlongTextures => "json_extract(t.result_json, '$.ulong_texture') = 1",
        AuditTarget::FragmentVaryings => {
            "coalesce(json_array_length(t.result_json, '$.fragment_varyings'), 0) > 0"
        }
        AuditTarget::LinkedFunctions => {
            "coalesce(json_array_length(t.result_json, '$.linked_functions'), 0) > 0"
        }
        AuditTarget::TileFunctions => "json_extract(t.result_json, '$.tile_function') = 1",
        AuditTarget::MeshDraws => "json_type(t.result_json, '$.mesh_draw') = 'object'",
    };
    let sql = format!(
        "SELECT s.air_sha256 FROM sources s JOIN triage_analysis t USING (air_sha256) \
         WHERE t.analyzer_abi=?1 AND (?2 IS NULL OR s.air_sha256>?2) AND {predicate} \
         ORDER BY s.air_sha256 LIMIT ?3"
    );
    let mut statement = connection
        .prepare(&sql)
        .map_err(|error| format!("prepare cached {target:?} audit query: {error}"))?;
    let hashes = statement
        .query_map(params![ANALYZER_ABI, after, limit as i64], |row| {
            row.get::<_, String>(0)
        })
        .map_err(|error| format!("query cached {target:?} audit: {error}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("read cached {target:?} audit: {error}"))?;
    Ok(hashes)
}

/// Select source identities that do not have facts under the current analyzer contract.
///
/// This is intentionally independent of case/review state: authored rows are part of the same
/// capability census as unplanned rows. The query touches only SQLite; callers read the returned
/// rows through their indexed byte locations.
pub fn select_uncached(index: &Path, limit: usize) -> Result<Vec<String>, String> {
    let connection = Connection::open(index)
        .map_err(|error| format!("open index {}: {error}", index.display()))?;
    ensure_cache_table(&connection)?;
    let mut statement = connection
        .prepare(
            "SELECT s.air_sha256 FROM sources s LEFT JOIN triage_analysis t USING (air_sha256) \
             WHERE t.air_sha256 IS NULL OR t.analyzer_abi<>?1 \
             ORDER BY s.air_sha256 LIMIT ?2",
        )
        .map_err(|error| format!("prepare uncached triage query: {error}"))?;
    let hashes = statement
        .query_map(params![ANALYZER_ABI, limit as i64], |row| {
            row.get::<_, String>(0)
        })
        .map_err(|error| format!("query uncached triage rows: {error}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("read uncached triage rows: {error}"))?;
    Ok(hashes)
}

/// Select every indexed source identity, irrespective of its triage-cache state.
///
/// This supports deliberate full regression sweeps without deleting the disposable index or
/// weakening the normal incremental path.
pub fn select_all(index: &Path, limit: usize) -> Result<Vec<String>, String> {
    let connection = Connection::open(index)
        .map_err(|error| format!("open index {}: {error}", index.display()))?;
    let mut statement = connection
        .prepare("SELECT air_sha256 FROM sources ORDER BY air_sha256 LIMIT ?1")
        .map_err(|error| format!("prepare full triage query: {error}"))?;
    let hashes = statement
        .query_map([limit as i64], |row| row.get::<_, String>(0))
        .map_err(|error| format!("query full triage rows: {error}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("read full triage rows: {error}"))?;
    Ok(hashes)
}

/// Summarize the exact current authoring-capability contract using only the disposable index.
pub fn authoring_capability_summary(index: &Path) -> Result<AuthoringCapabilitySummary, String> {
    let connection = Connection::open(index)
        .map_err(|error| format!("open index {}: {error}", index.display()))?;
    ensure_cache_table(&connection)?;
    let total_sources = connection
        .query_row("SELECT count(*) FROM sources", [], |row| row.get(0))
        .map_err(|error| format!("count authoring-capability sources: {error}"))?;
    let classified_sources = connection
        .query_row(
            "SELECT count(*) FROM sources s JOIN triage_analysis t USING (air_sha256) \
             WHERE t.analyzer_abi=?1",
            [ANALYZER_ABI],
            |row| row.get(0),
        )
        .map_err(|error| format!("count classified authoring-capability sources: {error}"))?;
    // Each list is the typed vocabulary's: names are snake_case ASCII, never input.
    let names = |requirements: &mut dyn Iterator<Item = ToolingRequirement>| {
        requirements
            .map(|requirement| format!("'{}'", requirement.as_str()))
            .collect::<Vec<_>>()
            .join(",")
    };
    let blocking = names(
        &mut ToolingRequirement::ALL
            .into_iter()
            .filter(|requirement| requirement.blocks_closure()),
    );
    let unresolved_sources = connection
        .query_row(
            &format!(
                "SELECT count(*) FROM sources s JOIN triage_analysis t USING (air_sha256) \
                 WHERE t.analyzer_abi=?1 \
                   AND (EXISTS (SELECT 1 FROM json_each(\
                            t.result_json, '$.tooling_requirements') requirement \
                        WHERE requirement.value IN ({blocking})) \
                        OR EXISTS (SELECT 1 FROM json_each(\
                            t.result_json, '$.unrecognized_air_intrinsics')))"
            ),
            [ANALYZER_ABI],
            |row| row.get(0),
        )
        .map_err(|error| format!("count unresolved authoring-capability sources: {error}"))?;
    let metal_limited = names(
        &mut ToolingRequirement::ALL
            .into_iter()
            .filter(|requirement| requirement.limit() == Limit::Metal),
    );
    let metal_refused_sources = connection
        .query_row(
            &format!(
                "SELECT count(*) FROM sources s JOIN triage_analysis t USING (air_sha256) \
                 WHERE t.analyzer_abi=?1 \
                   AND EXISTS (SELECT 1 FROM json_each(\
                            t.result_json, '$.tooling_requirements') requirement \
                        WHERE requirement.value IN ({metal_limited}))"
            ),
            [ANALYZER_ABI],
            |row| row.get(0),
        )
        .map_err(|error| format!("count sources Metal cannot build: {error}"))?;
    let mut executor_stops = BTreeMap::new();
    for executor in Executor::ALL {
        let stopping = |limit: Option<Limit>| {
            names(
                &mut ToolingRequirement::ALL.into_iter().filter(|requirement| {
                    requirement.stops_executor(executor)
                        && limit.is_none_or(|limit| requirement.limit() == limit)
                }),
            )
        };
        let with = |list: String| {
            format!(
                "EXISTS (SELECT 1 FROM json_each(t.result_json, '$.tooling_requirements') \
                 requirement WHERE requirement.value IN ({list}))"
            )
        };
        let stops = connection
            .query_row(
                &format!(
                    "SELECT count(*), coalesce(sum({}), 0), coalesce(sum({}), 0) \
                     FROM sources s JOIN triage_analysis t USING (air_sha256) \
                     WHERE t.analyzer_abi=?1 AND {}",
                    with(stopping(Some(Limit::Tooling))),
                    with(stopping(Some(Limit::Metal))),
                    with(stopping(None)),
                ),
                [ANALYZER_ABI],
                |row| {
                    Ok(ExecutorStops {
                        sources: row.get(0)?,
                        tooling_gap: row.get(1)?,
                        metal_refused: row.get(2)?,
                    })
                },
            )
            .map_err(|error| format!("count sources {executor} cannot run: {error}"))?;
        if stops.sources != 0 {
            executor_stops.insert(executor, stops);
        }
    }
    let mut statement = connection
        .prepare(
            "SELECT requirement.value, count(*) \
             FROM sources s JOIN triage_analysis t USING (air_sha256), \
                  json_each(t.result_json, '$.tooling_requirements') requirement \
             WHERE t.analyzer_abi=?1 \
             GROUP BY requirement.value ORDER BY requirement.value",
        )
        .map_err(|error| format!("prepare authoring-capability requirements: {error}"))?;
    let requirements = statement
        .query_map([ANALYZER_ABI], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, usize>(1)?))
        })
        .map_err(|error| format!("query authoring-capability requirements: {error}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("read authoring-capability requirements: {error}"))?
        .into_iter()
        .map(|(requirement, count)| Ok((requirement.parse()?, count)))
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    let mut statement = connection
        .prepare(
            "SELECT effect.value, count(DISTINCT s.air_sha256) \
             FROM sources s JOIN triage_analysis t USING (air_sha256), \
                  json_each(t.result_json, '$.unobserved_calls') effect \
             WHERE t.analyzer_abi=?1 \
             GROUP BY effect.value ORDER BY effect.value",
        )
        .map_err(|error| format!("prepare authoring-capability unobserved effects: {error}"))?;
    let unobserved_effects = statement
        .query_map([ANALYZER_ABI], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, usize>(1)?))
        })
        .map_err(|error| format!("query authoring-capability unobserved effects: {error}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("read authoring-capability unobserved effects: {error}"))?
        .into_iter()
        .map(|(effect, count)| Ok((effect.parse()?, count)))
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    let mut statement = connection
        .prepare(
            "SELECT intrinsic.key, sum(CAST(intrinsic.value AS INTEGER)) \
             FROM sources s JOIN triage_analysis t USING (air_sha256), \
                  json_each(t.result_json, '$.unrecognized_air_intrinsics') intrinsic \
             WHERE t.analyzer_abi=?1 \
             GROUP BY intrinsic.key ORDER BY intrinsic.key",
        )
        .map_err(|error| format!("prepare unrecognized AIR intrinsic summary: {error}"))?;
    let unrecognized_air_intrinsics = statement
        .query_map([ANALYZER_ABI], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, usize>(1)?))
        })
        .map_err(|error| format!("query unrecognized AIR intrinsic summary: {error}"))?
        .collect::<Result<BTreeMap<_, _>, _>>()
        .map_err(|error| format!("read unrecognized AIR intrinsic summary: {error}"))?;
    Ok(AuthoringCapabilitySummary {
        total_sources,
        classified_sources,
        unresolved_sources,
        metal_refused_sources,
        requirements,
        executor_stops,
        unobserved_effects,
        unrecognized_air_intrinsics,
    })
}

pub fn read_cached(
    index: &Path,
    hashes: &[String],
) -> Result<BTreeMap<String, StructuralTriage>, String> {
    let connection = Connection::open(index)
        .map_err(|error| format!("open index {}: {error}", index.display()))?;
    ensure_cache_table(&connection)?;
    let mut cached = BTreeMap::new();
    {
        let mut statement = connection
            .prepare(
                "SELECT air_sha256, analyzer_abi, result_json FROM triage_analysis \
                 WHERE air_sha256=?1 AND analyzer_abi=?2",
            )
            .map_err(|error| format!("prepare triage cache query: {error}"))?;
        for requested_hash in hashes {
            let row = statement
                .query_row(params![requested_hash, ANALYZER_ABI], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                })
                .optional()
                .map_err(|error| format!("query triage cache for {requested_hash}: {error}"))?;
            let Some((hash, _abi, json)) = row else {
                continue;
            };
            let mut result: StructuralTriage = serde_json::from_str(&json)
                .map_err(|error| format!("parse cached triage result for {hash}: {error}"))?;
            finalize_requirements(&mut result);
            cached.insert(hash, result);
        }
    }
    Ok(cached)
}

pub fn write_cached<'a>(
    index: &Path,
    results: impl IntoIterator<Item = (&'a str, &'a StructuralTriage)>,
) -> Result<(), String> {
    let mut connection = Connection::open(index)
        .map_err(|error| format!("open index {}: {error}", index.display()))?;
    ensure_cache_table(&connection)?;
    let transaction = connection
        .transaction()
        .map_err(|error| format!("begin triage cache update: {error}"))?;
    write_cached_transaction(&transaction, results)?;
    transaction
        .commit()
        .map_err(|error| format!("commit triage cache update: {error}"))
}

/// Recompute product-owned AIR disposition from cached exact call inventories.
///
/// This is the fast path for a lowering/recognition contract change: the structural analyzer has
/// already paid to parse every source and retained `air_calls` by exact ABI symbol. Reopening AIR
/// shards would make the index pointless. Keyset batches keep memory bounded independently of
/// corpus size and avoid holding a SQLite read cursor while committing each update batch.
pub fn reclassify_cached_air_intrinsics(index: &Path) -> Result<usize, String> {
    const BATCH: usize = 1024;
    let mut after = String::new();
    let mut updated = 0usize;
    loop {
        let rows = {
            let connection = Connection::open(index)
                .map_err(|error| format!("open index {}: {error}", index.display()))?;
            ensure_cache_table(&connection)?;
            let mut statement = connection
                .prepare(
                    "SELECT air_sha256, result_json FROM triage_analysis \
                     WHERE analyzer_abi=?1 AND air_sha256>?2 \
                     ORDER BY air_sha256 LIMIT ?3",
                )
                .map_err(|error| format!("prepare cached AIR reclassification: {error}"))?;
            let batch = statement
                .query_map(params![ANALYZER_ABI, after, BATCH as i64], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })
                .map_err(|error| format!("query cached AIR reclassification: {error}"))?
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| format!("read cached AIR reclassification: {error}"))?;
            batch
        };
        if rows.is_empty() {
            break;
        }
        after = rows.last().expect("nonempty batch").0.clone();
        let mut classified = Vec::with_capacity(rows.len());
        for (hash, json) in rows {
            let mut result: StructuralTriage = serde_json::from_str(&json)
                .map_err(|error| format!("decode cached triage row {hash}: {error}"))?;
            result.unrecognized_air_intrinsics =
                crate::executor_contract::unrecognized_air_intrinsics(&result.air_calls);
            result.unobserved_calls = crate::executor_contract::unobserved_calls(&result.air_calls);
            classified.push((hash, result));
        }
        write_cached(
            index,
            classified
                .iter()
                .map(|(hash, result)| (hash.as_str(), result)),
        )?;
        updated += classified.len();
    }
    Ok(updated)
}

fn ensure_cache_table(connection: &Connection) -> Result<(), String> {
    connection
        .execute_batch(
            "CREATE TABLE IF NOT EXISTS triage_analysis (
               air_sha256 TEXT PRIMARY KEY,
               analyzer_abi TEXT NOT NULL,
               result_json TEXT NOT NULL
             );",
        )
        .map_err(|error| format!("create triage cache: {error}"))
}

fn write_cached_transaction<'a>(
    transaction: &Transaction<'_>,
    results: impl IntoIterator<Item = (&'a str, &'a StructuralTriage)>,
) -> Result<(), String> {
    let mut insert = transaction
        .prepare(
            "INSERT INTO triage_analysis (air_sha256, analyzer_abi, result_json) \
             VALUES (?1, ?2, ?3) ON CONFLICT(air_sha256) DO UPDATE SET \
             analyzer_abi=excluded.analyzer_abi, result_json=excluded.result_json",
        )
        .map_err(|error| format!("prepare triage cache update: {error}"))?;
    for (hash, result) in results {
        let json = serde_json::to_string(result)
            .map_err(|error| format!("serialize triage result for {hash}: {error}"))?;
        insert
            .execute(params![hash, ANALYZER_ABI, json])
            .map_err(|error| format!("cache triage result for {hash}: {error}"))?;
    }
    Ok(())
}

/// The row classification reads for `source`: `source` itself when its module declares one entry,
/// and otherwise a row whose text is the entry's projection of the module
/// ([`crate::air::entry_text`]), so no fact of another entry is charged to this one. Every public
/// function here that derives a fact from a row reads it through this. A projection that fails is
/// an error naming the module and entry, never a module-wide fallback. A projection is read, never
/// keyed: its `air_sha256` stays the module's, so its own [`SourceRow::key`] is not the row's;
/// key by `source.key()`.
pub fn entry_row(source: &SourceRow) -> Result<Cow<'_, SourceRow>, String> {
    let text = crate::air::entry_text(&source.air_ll, &source.entry).map_err(|error| {
        format!(
            "{} ({}) has no projection onto its entry {}: {error}",
            source.label, source.air_sha256, source.entry
        )
    })?;
    Ok(match text {
        Cow::Borrowed(_) => Cow::Borrowed(source),
        Cow::Owned(air_ll) => Cow::Owned(SourceRow {
            air_sha256: source.air_sha256.clone(),
            stage: source.stage.clone(),
            entry: source.entry.clone(),
            air_ll,
            blob_b64: None,
            lib_sha256s: source.lib_sha256s.clone(),
            label: source.label.clone(),
        }),
    })
}

/// The facts of `source`'s entry ([`entry_row`]).
pub fn classify(source: &SourceRow) -> Result<StructuralTriage, String> {
    Ok(classify_inner(&*entry_row(source)?, true))
}

/// Classify corpus-wide support requirements without deriving the expensive per-call structural
/// signature. The blocker and tooling results are identical to [`classify`].
pub fn classify_summary(source: &SourceRow) -> Result<StructuralTriage, String> {
    Ok(classify_inner(&*entry_row(source)?, false))
}

fn classify_inner(source: &SourceRow, include_signature: bool) -> StructuralTriage {
    let cycle = has_cfg_cycle(&source.air_ll);
    let calls = crate::executor_contract::air_call_counts(&source.air_ll);
    let resource_markers = resource_marker_counts(&source.air_ll);
    let signature = if include_signature {
        structural_signature(source, cycle, &calls, &resource_markers)
    } else {
        String::new()
    };
    let unrecognized_air_intrinsics = crate::executor_contract::unrecognized_air_intrinsics(&calls);
    let reflection = reflect(source);
    let tooling_requirements = tooling_requirements(source, &calls, reflection.as_ref());
    let fragment_imageblock = reflection
        .as_ref()
        .is_some_and(|(_, reflection)| reflection.fragment_imageblock.is_some());
    let pipeline_state_members = reflection
        .as_ref()
        .is_some_and(|(_, reflection)| pipeline_state_fields(reflection).next().is_some());
    let tile_function = reflection.as_ref().is_some_and(|(stage, reflection)| {
        *stage == crate::case::Stage::Kernel
            && !reflection.implicit_imageblock_attachments.is_empty()
    });
    let mesh_draw = reflection.as_ref().and_then(|(_, reflection)| {
        crate::executor_contract::mesh::mesh_draw_facts(&source.air_ll, reflection)
    });
    let rasterization_rate_map = crate::executor_contract::rate_map::maps_coordinates(&calls);
    let sample_rate = crate::executor_contract::sample_rate::sample_rate_uses(
        &source.stage,
        &source.air_ll,
        &calls,
    );
    let ulong_texture = crate::executor_contract::calls_ulong_texture(&calls);
    let fragment_varyings = reflection
        .as_ref()
        .filter(|(stage, _)| *stage == crate::case::Stage::Fragment)
        .map(|(_, reflection)| fragment_varying_types(reflection))
        .unwrap_or_default();
    let linked_functions = linked_function_symbols(&source.air_ll);
    let uses_indirect_commands =
        crate::executor_contract::uses_indirect_commands(&source.air_ll, &calls);
    let unobserved_calls = crate::executor_contract::unobserved_calls(&calls);
    let mut result = StructuralTriage {
        signature,
        has_cfg_cycle: cycle,
        air_calls: calls,
        uses_indirect_commands,
        unobserved_calls,
        unrecognized_air_intrinsics,
        resource_markers,
        uses_device_addresses: source.air_ll.contains("inttoptr"),
        fragment_imageblock,
        pipeline_state_members,
        tile_function,
        mesh_draw,
        rasterization_rate_map,
        sample_rate,
        ulong_texture,
        fragment_varyings,
        linked_functions,
        tooling_requirements,
    };
    finalize_requirements(&mut result);
    result
}

/// [`StructuralTriage::linked_functions`] of an entry's text `ll`.
fn linked_function_symbols(ll: &str) -> Vec<String> {
    match crate::air::visible_references::reference_symbols(ll) {
        Ok(symbols) => symbols
            .into_iter()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect(),
        Err(error) => vec![format!("<{error}>")],
    }
}

/// [`StructuralTriage::fragment_varyings`] of a fragment function's `reflection`.
fn fragment_varying_types(reflection: &crate::air::reflect::ShaderInterface) -> Vec<String> {
    let mut varyings = reflection.varyings.iter().collect::<Vec<_>>();
    varyings.sort_by_key(|varying| varying.location);
    varyings
        .into_iter()
        .map(|varying| varying.type_name.clone().unwrap_or_else(|| "?".into()))
        .collect()
}

fn finalize_requirements(result: &mut StructuralTriage) {
    result.tooling_requirements.sort();
    result.tooling_requirements.dedup();
}

/// The declared interface of `source`'s entry ([`crate::air::reflect::inspect`], with no
/// function constants, as the old engine's `inspect` reflected a bare request) and the authored
/// stage it is, or `None` when the stage is unknown or the entry does not reflect.
fn reflect(
    source: &SourceRow,
) -> Option<(crate::case::Stage, crate::air::reflect::ShaderInterface)> {
    let stage = crate::case::Stage::from_metadata_label(&source.stage)?;
    crate::air::reflect::inspect(&source.air_ll, stage, &source.entry, &[])
        .ok()
        .map(|reflection| (stage, reflection))
}

fn tooling_requirements(
    source: &SourceRow,
    calls: &BTreeMap<String, usize>,
    reflection: Option<&(crate::case::Stage, crate::air::reflect::ShaderInterface)>,
) -> Vec<ToolingRequirement> {
    let mut requirements = crate::executor_contract::unsupported_air_requirements(calls)
        .into_iter()
        .collect::<Vec<_>>();
    // A mesh render pipeline's draw, from the entry's reflected structure with no partner named
    // (a row names none). An object or mesh row whose interface does not reflect has no stage a
    // case could be checked against, so every stage it needs is one nothing supplies.
    match reflection {
        Some((_, reflection)) => requirements.extend(
            crate::executor_contract::mesh::mesh_requirements(&source.air_ll, reflection, None),
        ),
        None if crate::case::Stage::from_metadata_label(&source.stage)
            .is_some_and(crate::case::Stage::is_mesh_pipeline_stage) =>
        {
            requirements.push(ToolingRequirement::MeshCompanion)
        }
        None => {}
    }
    // Outside the inspection gate: a module that only encodes commands is not resolved because the
    // product could not reflect it.
    requirements.extend(crate::executor_contract::indirect_command_requirement(
        &source.air_ll,
        calls,
        reflection.map(|(_, reflection)| reflection),
    ));
    if let Some((authored_stage, reflection)) = reflection {
        requirements.extend(crate::executor_contract::unsupported_source_requirements(
            &source.air_ll,
            reflection,
        ));
        requirements.extend(
            crate::executor_contract::unsupported_observation_requirements(
                *authored_stage,
                &source.air_ll,
                reflection,
            ),
        );
    }
    requirements
}

/// The argument-buffer members of `reflection` that hold a compute or render pipeline state.
fn pipeline_state_fields(
    reflection: &crate::air::reflect::ShaderInterface,
) -> impl Iterator<Item = &crate::air::reflect::EmbeddedArgBuffer> {
    use crate::air::reflect::ArgumentBufferResourceKind;
    reflection.argument_buffer_fields.iter().filter(|field| {
        matches!(
            field.resource_kind,
            Some(
                ArgumentBufferResourceKind::ComputePipelineState
                    | ArgumentBufferResourceKind::RenderPipelineState
            )
        )
    })
}

fn structural_signature(
    source: &SourceRow,
    cycle: bool,
    calls: &BTreeMap<String, usize>,
    resource_markers: &BTreeMap<String, usize>,
) -> String {
    let accesses = ["air.read", "air.write", "air.read_write", "air.sample"];
    let mut features = vec![
        format!("stage={}", source.stage),
        format!("cycle={cycle}"),
        format!(
            "defines={}",
            source
                .air_ll
                .lines()
                .filter(|line| line.trim_start().starts_with("define "))
                .count()
        ),
    ];
    for (kind, count) in resource_markers {
        features.push(format!("{kind}={count}"));
    }
    for access in accesses {
        features.push(format!(
            "{access}={}",
            source.air_ll.matches(access).count()
        ));
    }
    for (call, count) in calls {
        features.push(format!("call:{call}={count}"));
    }
    sha256_bytes(features.join("\n").as_bytes())
}

fn resource_marker_counts(ll: &str) -> BTreeMap<String, usize> {
    RESOURCE_KINDS
        .into_iter()
        .map(|kind| (kind.to_string(), ll.matches(kind).count()))
        .collect()
}

/// Where one source's coordinate-mapping calls read their rasterization-rate maps,
/// from the product's declared reflection and the shared placement
/// ([`crate::executor_contract::rate_map::rate_map_placement`]). `None` when the source calls no
/// coordinate-mapping function.
pub fn audit_rasterization_rate_map(source: &SourceRow) -> Result<Option<RateMapAudit>, String> {
    Ok(rasterization_rate_map(&*entry_row(source)?))
}

fn rasterization_rate_map(source: &SourceRow) -> Option<RateMapAudit> {
    use crate::executor_contract::rate_map::{maps_coordinates, rate_map_placement, MapPlace};
    if !maps_coordinates(&crate::executor_contract::air_call_counts(&source.air_ll)) {
        return None;
    }
    let Some((_, reflection)) = reflect(source) else {
        return Some(RateMapAudit {
            places: Vec::new(),
            unplaced: true,
            reflected: false,
        });
    };
    let placement = rate_map_placement(&source.air_ll, &reflection);
    let places = placement
        .places
        .iter()
        .map(|place| match place {
            MapPlace::Buffer { param } => format!("buffer:{param}"),
            MapPlace::ArgumentBufferMember { param, offset } => format!("member:{param}+{offset}"),
        })
        .collect();
    Some(RateMapAudit {
        places,
        unplaced: placement.unplaced,
        reflected: true,
    })
}

/// [`audit_rasterization_rate_map`]'s shape of one source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RateMapAudit {
    /// Each place a placed call reads a map from: `buffer:<param>` or
    /// `member:<param>+<byte offset>`, in order.
    pub places: Vec<String>,
    /// Some call reads its map from a place no literal covers.
    pub unplaced: bool,
    /// The product reflected the entry (without it, nothing is placed).
    pub reflected: bool,
}

impl RateMapAudit {
    pub fn shape(&self) -> String {
        let places = if self.places.is_empty() {
            "none".into()
        } else {
            self.places.join("+")
        };
        format!(
            "places={places},unplaced={},reflected={}",
            self.unplaced, self.reflected
        )
    }
}

#[cfg(test)]
pub(crate) mod classifier_inputs;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ScratchDir;

    #[test]
    fn only_unresolved_rows_keep_a_complete_census_open() {
        let summary = |total, classified, unresolved| AuthoringCapabilitySummary {
            total_sources: total,
            classified_sources: classified,
            unresolved_sources: unresolved,
            metal_refused_sources: 0,
            requirements: BTreeMap::new(),
            executor_stops: BTreeMap::new(),
            unobserved_effects: BTreeMap::new(),
            unrecognized_air_intrinsics: BTreeMap::new(),
        };
        // Closed when every row is classified and resolved.
        assert_eq!(summary(10, 10, 0).closure_gap(), None);
        assert_eq!(
            summary(10, 10, 3).closure_gap().as_deref(),
            Some("3 source(s) still have validation support gaps")
        );
        // A bounded batch that left rows unclassified gives no verdict.
        assert_eq!(summary(10, 9, 3).closure_gap(), None);
    }

    fn source(ll: &str) -> SourceRow {
        SourceRow {
            air_sha256: "11".repeat(32),
            stage: "Kernel".into(),
            entry: "main".into(),
            air_ll: ll.into(),
            blob_b64: None,
            lib_sha256s: vec!["22".repeat(32)],
            label: "local/test.ll".into(),
        }
    }

    #[test]
    fn cyclic_work_stays_actionable_without_entry_name_signatures() {
        let first =
            source("define void @first() {\nentry:\n br label %loop\nloop:\n br label %loop\n}");
        let mut renamed = first.clone();
        renamed.entry = "renamed".into();
        renamed.air_ll = renamed.air_ll.replace("@first", "@renamed");
        let first = classify(&first).unwrap();
        let renamed = classify(&renamed).unwrap();
        assert_eq!(first.signature, renamed.signature);
        assert!(first.tooling_requirements.is_empty());
    }

    #[test]
    fn unrecognized_air_families_are_first_class_product_gaps_with_exact_counts() {
        let row = source(
            "define void @main() {\nentry:\n %a = call i32 @air.future_tensor_op.i32(i32 1)\n %b = call i32 @air.future_tensor_op.i32(i32 2)\n ret void\n}\ndeclare i32 @air.future_tensor_op.i32(i32)",
        );
        let result = classify(&row).unwrap();
        assert_eq!(
            result.unrecognized_air_intrinsics,
            BTreeMap::from([("air.future_tensor_op.i32".into(), 2)])
        );
        assert!(result.tooling_requirements.is_empty());
    }

    #[test]
    fn recognized_and_linkage_intrinsics_do_not_create_unknown_family_gaps() {
        let row = source(
            "define void @main(ptr addrspace(1) %table) {\nentry:\n %a = call i32 @air.abs.s.i32(i32 -1)\n %b = call ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1) %table, i32 0)\n ret void\n}\ndeclare i32 @air.abs.s.i32(i32)\ndeclare ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1), i32)\n!0 = !{i32 0, !\"air.visible_function_table\"}",
        );
        let result = classify(&row).unwrap();
        assert!(result.unrecognized_air_intrinsics.is_empty());
        assert!(result.tooling_requirements.is_empty());
    }

    #[test]
    fn unused_visible_table_marker_is_already_authorable() {
        let row = source(
            "define void @main(ptr addrspace(1) %table) {\nentry:\n %f = call ptr @air.get_function_pointer_visible_function_table(ptr addrspace(1) %table, i32 7)\n ret void\n}\n!0 = !{i32 0, !\"air.visible_function_table\"}",
        );
        let result = classify(&row).unwrap();
        assert!(result.tooling_requirements.is_empty());
    }

    #[test]
    fn callback_free_intersection_table_is_not_a_separate_tooling_gap() {
        let row = source(
            "define void @main(ptr addrspace(1) %table) {\nentry:\n %hit = call { i32, float, i32, i32, ptr addrspace(1), i32, i32 } @air.intersect.instancing(<3 x float> zeroinitializer, <3 x float> zeroinitializer, float 0.0, float 1.0, ptr addrspace(1) null, i32 255, ptr addrspace(1) %table, ptr null, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 -1, i32 -1, i32 0, i1 false, i1 false)\n ret void\n}\n!0 = !{i32 0, !\"air.intersection_function_table\"}",
        );
        assert!(classify(&row).unwrap().tooling_requirements.is_empty());
    }

    #[test]
    fn callback_family_with_authored_opaque_table_is_not_a_tooling_gap() {
        let row = source(
            r#"define void @main(ptr addrspace(1) %table, ptr addrspace(1) %as) {
entry:
 %hit = call { i32, float, i32, i32, ptr addrspace(1), <2 x float>, i1 } @air.intersect.intersection_function_buffer.triangle_data(<3 x float> zeroinitializer, <3 x float> zeroinitializer, float 0.0, float 1.0, ptr addrspace(1) %as, ptr addrspace(1) %table, i64 1, i64 8, ptr null, i64 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 -1, i32 -1, i32 0, i1 false, i1 false)
 ret void
}
!air.kernel = !{!0}
!0 = !{ptr @main, !1, !2}
!1 = !{}
!2 = !{!3, !4}
!3 = !{i32 0, !"air.intersection_function_table", !"air.location_index", i32 6, i32 1, !"air.read"}
!4 = !{i32 1, !"air.primitive_acceleration_structure", !"air.location_index", i32 5, i32 1, !"air.read"}
"#,
        );
        assert!(classify(&row).unwrap().tooling_requirements.is_empty());
    }

    #[test]
    fn cached_requirement_selection_never_reads_source_bodies() {
        let scratch = ScratchDir::new("triage-requirement-selection").unwrap();
        let index = scratch.path().join("index.sqlite");
        let connection = Connection::open(&index).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE sources (air_sha256 TEXT PRIMARY KEY, stage TEXT NOT NULL);\n\
                 CREATE TABLE triage_analysis (\n\
                   air_sha256 TEXT PRIMARY KEY,\n\
                   analyzer_abi TEXT NOT NULL,\n\
                   result_json TEXT NOT NULL\n\
                 );\n\
                 INSERT INTO sources VALUES ('a', 'Kernel'), ('b', 'Kernel');",
            )
            .unwrap();
        let mut matching = classify(&source(
            "define void @main() { ret void }\n!0 = !{i32 0, !\"air.visible_function_table\"}",
        ))
        .unwrap();
        matching.tooling_requirements = vec![ToolingRequirement::IndirectCommandBuffer];
        let unrelated = classify(&source("define void @main() { ret void }")).unwrap();
        drop(connection);
        write_cached(&index, [("a", &matching), ("b", &unrelated)]).unwrap();
        assert_eq!(
            select_cached_requirement(&index, ToolingRequirement::IndirectCommandBuffer, 200)
                .unwrap(),
            ["a"]
        );
        assert!(select_cached_requirement_after(
            &index,
            ToolingRequirement::IndirectCommandBuffer,
            Some("a"),
            200,
        )
        .unwrap()
        .is_empty());
    }

    #[test]
    fn focused_audits_select_supported_structure_not_only_tooling_gaps() {
        let scratch = ScratchDir::new("triage-audit-targets").unwrap();
        let index = scratch.path().join("index.sqlite");
        let connection = Connection::open(&index).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE sources (air_sha256 TEXT PRIMARY KEY, stage TEXT NOT NULL);\n\
                 INSERT INTO sources VALUES ('device', 'Kernel'), ('ray', 'Kernel'), ('visible', 'Kernel'), ('command-call', 'Kernel'), ('command-resource', 'Kernel'), ('imageblock', 'Fragment'), ('pipelines', 'Kernel'), ('tile', 'Kernel'), ('rate-map', 'Kernel'), ('sample-rate', 'Fragment'), ('ulong', 'Kernel'), ('varyings', 'Fragment'), ('linked', 'Kernel'), ('mesh', 'Mesh');",
            )
            .unwrap();
        drop(connection);

        let visible = classify(&source(
            "define void @main() { ret void }\n!0 = !{i32 0, !\"air.visible_function_table\"}",
        ))
        .unwrap();
        let ray = classify(&source(
            "define void @main() { call void @air.intersect.synthetic() ret void }",
        ))
        .unwrap();
        let device = classify(&source(
            "define ptr @main(i64 %address) { %pointer = inttoptr i64 %address to ptr ret ptr %pointer }",
        )).unwrap();
        let mut command_call = classify(&source(
            "call void @air.set_barrier_compute_command(ptr addrspace(1) %commands, i32 0)",
        ))
        .unwrap();
        let mut command_resource = classify(&source("!0 = !{!\"air.command_buffer\"}")).unwrap();
        assert!(command_call.uses_indirect_commands);
        assert!(command_resource.uses_indirect_commands);
        assert_eq!(
            command_call.unobserved_calls,
            BTreeMap::from([(
                "air.set_barrier_compute_command".to_string(),
                UnobservedEffect::IndirectCommandContents
            )])
        );
        assert!(command_resource.unobserved_calls.is_empty());
        // The product cannot inspect this entry-less module, which leaves it encoding-only.
        assert_eq!(
            command_call.tooling_requirements,
            [ToolingRequirement::IndirectCommandBuffer]
        );
        assert!(command_resource.tooling_requirements.is_empty());
        // Gaining executor support must not remove either descriptive family from the audit.
        command_call.tooling_requirements.clear();
        command_resource.tooling_requirements.clear();
        assert!(visible.tooling_requirements.is_empty());
        // A fragment imageblock row: its family is described, and no requirement selects it.
        let mut imageblock_row = source(include_str!(
            "../fixtures/public/fragment_custom_imageblock.ll"
        ));
        imageblock_row.stage = "Fragment".into();
        imageblock_row.entry = "fragment_custom_imageblock".into();
        let imageblock = classify(&imageblock_row).unwrap();
        assert!(imageblock.fragment_imageblock);
        assert!(imageblock.tooling_requirements.is_empty());
        // A row holding pipeline states: its family is described, and no requirement selects it.
        let pipelines = classify(&source(&pipeline_state_kernel("main"))).unwrap();
        assert!(pipelines.pipeline_state_members);
        assert!(pipelines.tooling_requirements.is_empty());
        // A tile function row: described, and no requirement selects it.
        let mut tile_row = source(include_str!(
            "../fixtures/public/kernel_implicit_imageblock_three_widths.ll"
        ));
        tile_row.stage = "Kernel".into();
        tile_row.entry = "kernel_implicit_imageblock_three_widths".into();
        let tile = classify(&tile_row).unwrap();
        assert!(tile.tile_function);
        assert!(!imageblock.tile_function);
        assert!(tile.tooling_requirements.is_empty());
        // A row that maps coordinates through rate maps a pointer array holds: described, and no
        // requirement selects it.
        let mut rate_map_row = source(include_str!(
            "../fixtures/public/kernel_rate_map_pointer_array.ll"
        ));
        rate_map_row.entry = "kernel_rate_map_pointer_array".into();
        let rate_map = classify(&rate_map_row).unwrap();
        assert!(rate_map.rasterization_rate_map);
        assert!(!tile.rasterization_rate_map);
        assert!(rate_map.tooling_requirements.is_empty());
        // A row that reads a sample-rate feature: described, and no requirement selects it.
        let mut sample_rate_row = source(include_str!("../fixtures/public/engine/sample_rate.ll"));
        sample_rate_row.stage = "Fragment".into();
        sample_rate_row.entry = "sample_rate_mask_in".into();
        let sample_rate = classify(&sample_rate_row).unwrap();
        assert_eq!(
            sample_rate.sample_rate,
            [
                crate::executor_contract::sample_rate::SampleRateUse::SampleId,
                crate::executor_contract::sample_rate::SampleRateUse::SampleMaskIn
            ]
        );
        assert!(imageblock.sample_rate.is_empty() && rate_map.sample_rate.is_empty());
        assert!(sample_rate.tooling_requirements.is_empty());
        // A row that operates on a `ulong` texture: described, and no executor's gap.
        let mut ulong_row = source(include_str!(
            "../fixtures/public/engine/unk_ulong_atomics.ll"
        ));
        ulong_row.entry = "unk_ulong_atomic_rw".into();
        let ulong = classify(&ulong_row).unwrap();
        assert!(ulong.ulong_texture);
        assert!(ulong.tooling_requirements.is_empty());
        assert!(!sample_rate.ulong_texture && !tile.ulong_texture);
        // A fragment function that reads a varying: described with the varying's declared type,
        // and no requirement selects it. The fragment functions above read none.
        let mut varyings_row = source(include_str!("../fixtures/public/fragment_varying_color.ll"));
        varyings_row.stage = "Fragment".into();
        varyings_row.entry = "fragment_varying_color".into();
        let varyings = classify(&varyings_row).unwrap();
        assert_eq!(varyings.fragment_varyings, ["float2"]);
        assert!(varyings.tooling_requirements.is_empty());
        assert!(
            imageblock.fragment_varyings.is_empty() && sample_rate.fragment_varyings.is_empty()
        );
        // An entry that calls visible functions another module defines: described with the
        // symbols its own projection references, and no requirement selects it. The caller's other
        // kernel references another symbol, which is not this entry's.
        let caller_ll = include_str!("../fixtures/public/engine/linked_visible_caller.ll");
        let mut linked_row = source(caller_ll);
        linked_row.entry = "linked_visible_caller".into();
        let linked = classify(&linked_row).unwrap();
        assert_eq!(linked.linked_functions, ["linked_blend", "linked_scale"]);
        assert!(linked.tooling_requirements.is_empty());
        let mut sampling_row = source(caller_ll);
        sampling_row.entry = "linked_visible_sampling".into();
        assert_eq!(
            classify(&sampling_row).unwrap().linked_functions,
            ["linked_sample"]
        );
        assert!(visible.linked_functions.is_empty() && varyings.linked_functions.is_empty());
        let mesh = classify(&entry_of("engine/mesh_draws.ll", "mesh_draws_mesh")).unwrap();
        assert!(mesh.mesh_draw.is_some() && tile.mesh_draw.is_none());
        write_cached(
            &index,
            [
                ("visible", &visible),
                ("ray", &ray),
                ("device", &device),
                ("command-call", &command_call),
                ("command-resource", &command_resource),
                ("imageblock", &imageblock),
                ("pipelines", &pipelines),
                ("tile", &tile),
                ("rate-map", &rate_map),
                ("sample-rate", &sample_rate),
                ("ulong", &ulong),
                ("varyings", &varyings),
                ("linked", &linked),
                ("mesh", &mesh),
            ],
        )
        .unwrap();
        assert_eq!(
            select_cached_audit_target_after(&index, AuditTarget::MeshDraws, None, 10).unwrap(),
            ["mesh"]
        );
        assert_eq!(
            select_cached_audit_target_after(&index, AuditTarget::LinkedFunctions, None, 10)
                .unwrap(),
            ["linked"]
        );
        assert_eq!(
            select_cached_audit_target_after(&index, AuditTarget::TileFunctions, None, 10).unwrap(),
            ["tile"]
        );
        assert_eq!(
            select_cached_audit_target_after(&index, AuditTarget::FragmentVaryings, None, 10)
                .unwrap(),
            ["varyings"]
        );
        assert_eq!(
            select_cached_audit_target_after(&index, AuditTarget::SampleRate, None, 10).unwrap(),
            ["sample-rate"]
        );
        assert_eq!(
            select_cached_audit_target_after(&index, AuditTarget::UlongTextures, None, 10).unwrap(),
            ["ulong"]
        );
        assert_eq!(
            select_cached_audit_target_after(&index, AuditTarget::RasterizationRateMaps, None, 10)
                .unwrap(),
            ["rate-map"]
        );
        assert_eq!(
            select_cached_audit_target_after(&index, AuditTarget::IndirectCommands, None, 10)
                .unwrap(),
            ["command-call", "command-resource"]
        );
        assert_eq!(
            select_cached_audit_target_after(
                &index,
                AuditTarget::IndirectCommands,
                Some("command-call"),
                1
            )
            .unwrap(),
            ["command-resource"]
        );
    }

    #[test]
    fn authoring_capability_census_includes_every_source_state() {
        let scratch = ScratchDir::new("authoring-capability-census").unwrap();
        let index = scratch.path().join("index.sqlite");
        let connection = Connection::open(&index).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE sources (air_sha256 TEXT PRIMARY KEY, stage TEXT NOT NULL);\n\
                 INSERT INTO sources VALUES \
                    ('authored', 'Kernel'), ('reviewed', 'Kernel'), \
                    ('unknown-air', 'Kernel'), ('unplanned', 'Kernel');",
            )
            .unwrap();
        drop(connection);

        let supported = classify(&source("define void @main() { ret void }")).unwrap();
        let mut unsupported = supported.clone();
        unsupported.tooling_requirements = vec![ToolingRequirement::IndirectCommandBuffer];
        // Unobserved effects are reported, but never make a source unresolved.
        let mut encoding = supported.clone();
        encoding.unobserved_calls = BTreeMap::from([
            (
                "air.set_barrier_compute_command".into(),
                UnobservedEffect::IndirectCommandContents,
            ),
            (
                "air.set_kernel_buffer_compute_command.p1i8".into(),
                UnobservedEffect::IndirectCommandContents,
            ),
        ]);
        unsupported.unobserved_calls = encoding.unobserved_calls.clone();
        let mut unknown_air = supported.clone();
        unknown_air.unrecognized_air_intrinsics =
            BTreeMap::from([("air.future_tensor.i32".into(), 3)]);
        write_cached(
            &index,
            [
                ("authored", &encoding),
                ("reviewed", &unsupported),
                ("unknown-air", &unknown_air),
            ],
        )
        .unwrap();

        assert_eq!(select_uncached(&index, 100).unwrap(), ["unplanned"]);
        assert_eq!(select_uncached(&index, 0).unwrap(), Vec::<String>::new());
        assert_eq!(
            select_all(&index, 100).unwrap(),
            ["authored", "reviewed", "unknown-air", "unplanned"]
        );
        assert_eq!(select_all(&index, 2).unwrap(), ["authored", "reviewed"]);
        let summary = authoring_capability_summary(&index).unwrap();
        assert_eq!(summary.total_sources, 4);
        assert_eq!(summary.classified_sources, 3);
        assert_eq!(summary.remaining_sources(), 1);
        assert_eq!(summary.unresolved_sources, 2);
        assert_eq!(
            summary.requirements[&ToolingRequirement::IndirectCommandBuffer],
            1
        );
        assert_eq!(
            summary.unobserved_effects,
            BTreeMap::from([(UnobservedEffect::IndirectCommandContents, 2)])
        );
        assert_eq!(
            summary.unrecognized_air_intrinsics["air.future_tensor.i32"],
            3
        );
        assert_eq!(
            summary.executor_stops,
            Executor::ALL
                .into_iter()
                .map(|executor| {
                    let stops = ExecutorStops {
                        sources: 1,
                        tooling_gap: 1,
                        metal_refused: 0,
                    };
                    (executor, stops)
                })
                .collect()
        );
    }

    /// A row is unresolved when a requirement stops an executor the census closes on (the Metal
    /// oracle or the engine's candidate), and each executor counts the rows it cannot run. A
    /// requirement whose limit is Metal's own stops every executor but closes its row, counted as
    /// refused by Metal; a tooling gap beside it still keeps that row open.
    #[test]
    fn the_census_closes_on_the_metal_oracle_and_the_engine() {
        let scratch = ScratchDir::new("authoring-capability-executors").unwrap();
        let index = scratch.path().join("index.sqlite");
        let connection = Connection::open(&index).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE sources (air_sha256 TEXT PRIMARY KEY, stage TEXT NOT NULL);\n\
                 INSERT INTO sources VALUES \
                    ('new-only', 'Fragment'), ('authoring', 'Vertex'), ('none', 'Kernel'), \
                    ('metal-refused', 'Kernel'), ('metal-refused-and-gap', 'Kernel');",
            )
            .unwrap();
        drop(connection);
        let supported = classify(&source("define void @main() { ret void }")).unwrap();
        let with = |requirements: &[ToolingRequirement]| StructuralTriage {
            tooling_requirements: requirements.to_vec(),
            ..supported.clone()
        };
        let new_only = with(&[ToolingRequirement::RenderDeviceBufferArray]);
        let authoring = with(&[ToolingRequirement::VertexSideEffectObservation]);
        let metal_refused = with(&[ToolingRequirement::KernelMinLodClamp]);
        let metal_refused_and_gap = with(&[
            ToolingRequirement::KernelMinLodClamp,
            ToolingRequirement::RenderDeviceBufferArray,
        ]);
        write_cached(
            &index,
            [
                ("new-only", &new_only),
                ("authoring", &authoring),
                ("none", &supported),
                ("metal-refused", &metal_refused),
                ("metal-refused-and-gap", &metal_refused_and_gap),
            ],
        )
        .unwrap();
        let summary = authoring_capability_summary(&index).unwrap();
        assert_eq!(summary.classified_sources, 5);
        assert_eq!(summary.unresolved_sources, 3);
        assert_eq!(summary.metal_refused_sources, 2);
        let stops = |sources, tooling_gap, metal_refused| ExecutorStops {
            sources,
            tooling_gap,
            metal_refused,
        };
        assert_eq!(
            summary.executor_stops,
            BTreeMap::from([
                (Executor::MetalOracle, stops(3, 1, 2)),
                (Executor::EngineCandidate, stops(4, 3, 2)),
            ])
        );
        assert_eq!(
            summary.closure_gap().as_deref(),
            Some("3 source(s) still have validation support gaps")
        );
    }

    #[test]
    fn indirect_command_fixtures_close_beside_another_observable_effect_or_through_replay() {
        fn fixture(name: &str) -> SourceRow {
            let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("fixtures/public")
                .join(name);
            let air_ll = std::fs::read_to_string(&path).unwrap();
            let (stage, entry) = crate::air::stage_entry_from_ll(&air_ll).unwrap();
            SourceRow {
                stage: stage.into(),
                entry,
                ..source(&air_ll)
            }
        }
        let unobserved = |symbols: &[&str]| {
            symbols
                .iter()
                .map(|symbol| {
                    (
                        symbol.to_string(),
                        UnobservedEffect::IndirectCommandContents,
                    )
                })
                .collect::<BTreeMap<_, _>>()
        };

        let declared = classify(&fixture("kernel_icb_declared_handle_is_inert.ll")).unwrap();
        assert!(declared.uses_indirect_commands);
        assert!(declared.unobserved_calls.is_empty());
        assert!(declared.tooling_requirements.is_empty());

        let compute = fixture("kernel_icb_encode_compute_commands.ll");
        let classified = classify(&compute).unwrap();
        assert!(classified.uses_indirect_commands);
        assert_eq!(
            classified.unobserved_calls,
            unobserved(&[
                "air.concurrent_dispatch_threadgroups_compute_command",
                "air.set_barrier_compute_command",
                "air.set_kernel_buffer_compute_command.p1i8",
                "air.set_threadgroup_memory_length_compute_command",
            ])
        );
        assert!(classified.tooling_requirements.is_empty());

        let render = classify(&fixture("kernel_icb_encode_render_commands.ll")).unwrap();
        assert_eq!(
            render.unobserved_calls,
            unobserved(&[
                "air.draw_primitives_render_command",
                "air.set_fragment_buffer_render_command.p1i8",
                "air.set_vertex_buffer_render_command.p1i8",
            ])
        );
        assert!(render.tooling_requirements.is_empty());

        let encoding_only = SourceRow {
            air_ll: compute
                .air_ll
                .lines()
                .filter(|line| !line.trim_start().starts_with("store "))
                .collect::<Vec<_>>()
                .join("\n"),
            ..compute
        };
        let classified = classify(&encoding_only).unwrap();
        assert_eq!(classified.unobserved_calls.len(), 4);
        // A kernel's compute commands are observed by executing them, which both executors do: no
        // requirement, and a case of the module must execute them.
        let commands_only = |row: &SourceRow| {
            crate::executor_contract::commands_only(
                &row.air_ll,
                &crate::executor_contract::air_call_counts(&row.air_ll),
                reflect(row).as_ref().map(|(_, reflection)| reflection),
            )
        };
        use crate::executor_contract::CommandsOnly;
        assert!(classified.tooling_requirements.is_empty());
        assert_eq!(commands_only(&encoding_only), Some(CommandsOnly::Replayed));
        // A kernel's render commands are observed by executing them too: its vertex buffer and draw
        // select what a replayed draw writes.
        let render = fixture("kernel_icb_encode_render_commands.ll");
        let render_only = SourceRow {
            air_ll: render
                .air_ll
                .lines()
                .filter(|line| !line.trim_start().starts_with("store "))
                .collect::<Vec<_>>()
                .join("\n"),
            ..render
        };
        assert!(classify(&render_only)
            .unwrap()
            .tooling_requirements
            .is_empty());
        assert_eq!(commands_only(&render_only), Some(CommandsOnly::Replayed));
        // With its vertex buffer and draw gone, its fragment buffer alone selects nothing a
        // replayed command writes, and it keeps the command-buffer requirement.
        let fragment_only = SourceRow {
            air_ll: render_only
                .air_ll
                .lines()
                .filter(|line| {
                    !line.contains("call void @air.set_vertex_buffer_render_command")
                        && !line.contains("call void @air.draw_primitives_render_command")
                })
                .collect::<Vec<_>>()
                .join("\n"),
            ..render_only.clone()
        };
        assert_eq!(
            classify(&fragment_only).unwrap().tooling_requirements,
            [ToolingRequirement::IndirectCommandBuffer]
        );
        assert_eq!(
            commands_only(&fragment_only),
            Some(CommandsOnly::Unobserved)
        );
        // The owned render replay fixtures: a rasterization-disabled vertex function whose only
        // effect is the vertex and fragment buffers it encodes, and a kernel whose only effect is
        // its draws, are observed through replay; the template they run writes memory itself.
        for producer in [
            "engine/icb_render_replay_encode.ll",
            "engine/icb_render_replay_encode_draw_only.ll",
        ] {
            let row = fixture(producer);
            let classified = classify(&row).unwrap();
            assert!(classified.uses_indirect_commands, "{producer}");
            assert!(classified.tooling_requirements.is_empty(), "{producer}");
            assert_eq!(
                commands_only(&row),
                Some(CommandsOnly::Replayed),
                "{producer}"
            );
        }
        let template = classify(&fixture("engine/icb_render_replay_template.ll")).unwrap();
        assert!(!template.uses_indirect_commands);
        assert!(template.tooling_requirements.is_empty());
        // The same compute module with no kernel list: the product's inspection selects no
        // entry, and the row keeps the command-buffer requirement rather than counting as
        // resolved or replayable.
        let uninspectable = SourceRow {
            air_ll: encoding_only
                .air_ll
                .lines()
                .map(|line| {
                    if line.starts_with("!air.kernel = ") {
                        "!air.kernel = !{}"
                    } else {
                        line
                    }
                })
                .collect::<Vec<_>>()
                .join("\n"),
            ..encoding_only.clone()
        };
        assert!(crate::air::reflect::inspect(
            uninspectable.air_ll.as_str(),
            crate::case::Stage::Kernel,
            &uninspectable.entry,
            &[]
        )
        .is_err());
        assert_eq!(
            classify(&uninspectable).unwrap().tooling_requirements,
            [ToolingRequirement::IndirectCommandBuffer]
        );
    }

    #[test]
    fn acceleration_structure_contract_is_actionable_not_a_review_blocker() {
        let row = source(
            "define void @main(ptr addrspace(1) %as) { ret void }\n!0 = !{i32 0, !\"air.instance_acceleration_structure\"}",
        );
        let result = classify(&row).unwrap();
        assert!(result.tooling_requirements.is_empty());
    }

    #[test]
    fn callback_free_primitive_intersection_is_supported_end_to_end() {
        let resource_only = source(
            "define void @main(ptr addrspace(1) %as) { ret void }\n!0 = !{i32 0, !\"air.primitive_acceleration_structure\"}",
        );
        assert!(classify(&resource_only)
            .unwrap()
            .tooling_requirements
            .is_empty());

        let intersect = source(
            "define void @main(ptr addrspace(1) %as) { %hit = call i1 @air.intersect.triangle_data(ptr addrspace(1) %as) ret void }\n!0 = !{i32 0, !\"air.primitive_acceleration_structure\"}",
        );
        assert!(classify(&intersect)
            .unwrap()
            .tooling_requirements
            .is_empty());
    }

    #[test]
    fn multi_level_device_address_walk_is_supported_end_to_end() {
        let row = source(
            "define void @main(ptr addrspace(1) %as, ptr addrspace(1) %table, i64 %address, ptr %ids, ptr %user_ids) {\nentry:\n %hit = call { i32, float, i32, i32, ptr addrspace(1), i8 } @air.intersect.multi_level_instancing(<3 x float> zeroinitializer, <3 x float> zeroinitializer, float 0.0, float 1.0, ptr addrspace(1) %as, i32 255, ptr addrspace(1) %table, ptr null, i64 0, i8 2, ptr %ids, ptr %user_ids, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 -1, i32 -1, i32 0, i1 false, i1 false)\n %child = inttoptr i64 %address to ptr addrspace(1)\n ret void\n}",
        );
        assert!(classify(&row).unwrap().tooling_requirements.is_empty());
    }

    #[test]
    fn tooling_requirements_are_independent_not_precedence_ordered() {
        let mut row = source(
            "define void @main(ptr addrspace(1) %c) {\n  call void @air.set_object_buffer_render_command.p1i8(ptr addrspace(1) %c, i32 0, ptr addrspace(1) %c, i64 -1, i32 0)\n  ret void\n}\n!0 = !{i32 0, !\"air.imageblock\"}\n!1 = !{i32 1, !\"air.command_buffer\", !\"air.location_index\", i32 0}\n!2 = !{i32 2, !\"air.texture\", !\"air.arg_type_name\", !\"array_ref<texture2d<float, sample>>\"}",
        );
        row.stage = "Fragment".into();
        let result = classify(&row).unwrap();
        assert_eq!(
            result.tooling_requirements,
            [ToolingRequirement::IndirectCommandBuffer]
        );
    }

    /// A coordinate-mapping call's map at offset 0 of a buffer is the literal's to place; map data
    /// at another offset stays bytes the engine's candidate cannot run.
    #[test]
    fn rate_map_data_away_from_offset_0_is_a_engine_requirement() {
        let fixture = include_str!("../fixtures/public/kernel_rate_map_physical_to_screen.ll");
        let row = |ll: &str| {
            let mut row = source(ll);
            row.entry = "kernel_rate_map_physical_to_screen".into();
            row
        };
        let placed = classify(&row(fixture)).unwrap();
        assert!(
            !placed
                .tooling_requirements
                .contains(&ToolingRequirement::RasterizationRateMapPlacement),
            "{:?}",
            placed.tooling_requirements
        );
        let offset = fixture.replace(
            "%7 = bitcast ptr addrspace(2) %0 to ptr addrspace(2)",
            "%7 = getelementptr inbounds i8, ptr addrspace(2) %0, i64 16",
        );
        assert_ne!(offset, fixture);
        let unplaced = classify(&row(&offset)).unwrap();
        // The family stays described whether or not its maps are placed.
        assert!(placed.rasterization_rate_map && unplaced.rasterization_rate_map);
        assert_eq!(
            audit_rasterization_rate_map(&row(fixture))
                .unwrap()
                .unwrap()
                .shape(),
            "places=buffer:0,unplaced=false,reflected=true"
        );
        assert_eq!(
            audit_rasterization_rate_map(&row(&offset))
                .unwrap()
                .unwrap()
                .shape(),
            "places=none,unplaced=true,reflected=true"
        );
        assert_eq!(
            unplaced
                .tooling_requirements
                .iter()
                .copied()
                .filter(|requirement| !placed.tooling_requirements.contains(requirement))
                .collect::<Vec<_>>(),
            [ToolingRequirement::RasterizationRateMapPlacement]
        );
        assert!(ToolingRequirement::RasterizationRateMapPlacement.blocks_closure());
    }

    #[test]
    fn unknown_implicit_imageblock_suffix_is_not_a_false_clean_census_row() {
        let supported = source(
            "define void @main() { %v = call <2 x half> @air.load.implicit_imageblock.v2f16(i32 0, <2 x i16> zeroinitializer, i32 0, i16 0) ret void }\ndeclare <2 x half> @air.load.implicit_imageblock.v2f16(i32, <2 x i16>, i32, i16)",
        );
        assert!(classify(&supported)
            .unwrap()
            .tooling_requirements
            .is_empty());
        // A `float2` color's plane is `RG32Float`, which a case authors.
        let float2 = source(
            "define void @main() { %v = call <2 x float> @air.load.implicit_imageblock.v2f32(i32 0, <2 x i16> zeroinitializer, i32 0, i16 0) ret void }\ndeclare <2 x float> @air.load.implicit_imageblock.v2f32(i32, <2 x i16>, i32, i16)",
        );
        assert!(classify(&float2).unwrap().tooling_requirements.is_empty());

        let unknown = source(
            "define void @main() { %v = call <3 x half> @air.load.implicit_imageblock.v3f16(i32 0, <2 x i16> zeroinitializer, i32 0, i16 0) ret void }\ndeclare <3 x half> @air.load.implicit_imageblock.v3f16(i32, <2 x i16>, i32, i16)",
        );
        assert_eq!(
            classify(&unknown).unwrap().tooling_requirements,
            [ToolingRequirement::ImplicitImageblockLiteral]
        );
    }

    /// A tile function's multisample operations (the pixel's sample count, colors and their
    /// coverage masks, a store to the samples of a mask) are named: the product refuses them by
    /// feature, so the row states no tooling requirement and no unrecognized intrinsic, and its
    /// masked store still reflects its attachment's plane.
    #[test]
    fn multisample_tile_operations_are_named_product_refusals_not_tooling_gaps() {
        let tile = source(
            "define void @main() {\n\
             %s = call i16 @air.get_imageblock_samples()\n\
             %n = call i16 @air.get_imageblock_num_colors(<2 x i16> zeroinitializer)\n\
             %c = call i16 @air.get_color_coverage_mask(<2 x i16> zeroinitializer, i16 0)\n\
             call void @air.store.implicit_imageblock.mask.v4f16(<4 x half> zeroinitializer, i32 0, <2 x i16> zeroinitializer, i16 %c)\n\
             ret void\n\
             }\n\
             declare i16 @air.get_imageblock_samples()\n\
             declare i16 @air.get_imageblock_num_colors(<2 x i16>)\n\
             declare i16 @air.get_color_coverage_mask(<2 x i16>, i16)\n\
             declare void @air.store.implicit_imageblock.mask.v4f16(<4 x half>, i32, <2 x i16>, i16)\n\
             !air.kernel = !{!0}\n\
             !0 = !{ptr @main, !1, !1}\n\
             !1 = !{}",
        );
        let triage = classify(&tile).unwrap();
        assert!(triage.tooling_requirements.is_empty());
        assert!(triage.unrecognized_air_intrinsics.is_empty());
        assert!(triage.tile_function);
    }

    /// Cache validity follows semantics: the key is the classifier's version, the product
    /// fingerprint, and the fingerprint of the classifier's own sources, hashed the way
    /// `validation/build.rs` hashes them. An edit to any of them changes the key; an edit to other
    /// validation code does not.
    #[test]
    fn analyzer_key_covers_the_classifier_version_product_and_sources_only() {
        use sha2::{Digest, Sha256};
        use std::path::Path;
        let analyzer_abi = |version: &str, product: &str, classifier: &str| {
            format!("{version}:{product}:{classifier}")
        };
        assert_eq!(
            ANALYZER_ABI,
            analyzer_abi(
                analyzer_version!(),
                env!("METAL2VULKAN_PRODUCT_FINGERPRINT"),
                env!("METAL2VULKAN_CLASSIFIER_FINGERPRINT")
            )
        );
        // The build's fingerprints are the product's and the classifier's own sources, hashed as
        // below.
        let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));

        let inputs = env!("METAL2VULKAN_CLASSIFIER_INPUTS")
            .split(':')
            .collect::<Vec<_>>();
        let product = crate::product_fingerprint::product_fingerprint;
        let classifier = |validation: &Path| {
            let files = inputs
                .iter()
                .map(|path| validation.join(path))
                .collect::<Vec<_>>();
            let mut hash = Sha256::new();
            crate::build_support::hash_files(&mut hash, validation, &files);
            crate::hash::hex_encode(hash.finalize())
        };
        assert_eq!(
            product(&crate_root.join("..")),
            env!("METAL2VULKAN_PRODUCT_FINGERPRINT")
        );
        assert_eq!(
            classifier(crate_root),
            env!("METAL2VULKAN_CLASSIFIER_FINGERPRINT"),
            "the test hashes the classifier's sources the way the build does"
        );

        // The inputs are closed over what their non-test code reaches in this crate: every module
        // file a path or `mod` declaration resolves to is an input, or is listed here with why
        // classification does not depend on it.
        let not_classification = BTreeMap::from([
            (
                "src/air/visible_references.rs",
                "the checker's closure of a case's authored linked modules; classification reads \
                 no case",
            ),
            (
                "src/source.rs",
                "the SourceRow record classification receives; its fields are data, not rules",
            ),
            (
                "src/case/indirect_commands.rs",
                "the authored ICB literal schema; classification reads no case literal",
            ),
            (
                "src/case/out_of_bounds.rs",
                "the authored out-of-bounds schema; classification reads no case literal",
            ),
            (
                "src/jsonl.rs",
                "case shard serialization; classification reads no shard",
            ),
            (
                "src/literal.rs",
                "case literal decoding; classification decodes no literal",
            ),
            (
                "src/executor_contract/buffer_slots.rs",
                "the executors' buffer-slot rule; classification binds no buffer",
            ),
            (
                "src/executor_contract/multisample.rs",
                "the executors' per-sample attachment transfer and readback; classification fills \
                 and reads no attachment",
            ),
            (
                "src/executor_contract/out_of_bounds.rs",
                "the executors' out-of-bounds containment verdict; classification runs no case",
            ),
            (
                "src/case/sparse.rs",
                "the authored sparse-texture schema and its checks; classification reads no case \
                 literal",
            ),
            (
                "src/case/mips.rs",
                "the authored mip-level schema, its checks, and the levels' literal layout; \
                 classification reads no case literal",
            ),
            (
                "src/executor_contract/sparse.rs",
                "the executors' sparse tile mappings; classification maps no tile",
            ),
            (
                "src/executor_contract/depth_stencil.rs",
                "the executors' depth-stencil state; classification creates no pipeline",
            ),
            (
                "src/executor_contract/depth_range.rs",
                "the executors' depth-range rule over a case's depth bytes and the device's \
                 VK_EXT_depth_range_unrestricted; classification reads no depth and opens no \
                 device",
            ),
            (
                "src/executor_contract/render_target_array.rs",
                "the executors' render target array length; classification draws no pass",
            ),
            (
                "src/executor_contract/stage_inputs.rs",
                "the executors' kernel stage-input fetch; classification fetches no attribute",
            ),
            (
                "src/executor_contract/tessellation.rs",
                "the executors' tessellation pipeline state and patch draw; classification draws \
                 no patch",
            ),
            (
                "src/budgets.rs",
                "the shared pipeline-creation ceiling, which the case checker holds a case's own \
                 creation bound above and the executors fall back to; classification checks no \
                 case and makes no pipeline",
            ),
        ]);
        let dependencies =
            super::classifier_inputs::dependencies(crate_root, &[], &inputs).unwrap();
        let unlisted = dependencies
            .iter()
            .filter(|(file, _)| {
                !inputs.contains(&file.as_str()) && !not_classification.contains_key(file.as_str())
            })
            .collect::<Vec<_>>();
        assert!(
            unlisted.is_empty(),
            "classifier inputs reach files that are neither inputs nor listed: {unlisted:?}"
        );
        for file in not_classification.keys() {
            assert!(
                dependencies.contains_key(*file),
                "{file} is listed but no classifier input reaches it"
            );
        }

        // A synthetic crates/ directory, fingerprinted the way this crate's build.rs fingerprints it.
        let scratch = ScratchDir::new("analyzer-key").unwrap();
        let root = scratch.path();
        let write = |path: &str, text: &str| {
            let path = root.join(path);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, text).unwrap();
        };
        write("metal2vulkan/Cargo.toml", "[package]\n");
        write("metal2vulkan/build.rs", "fn main() {}\n");
        write("metal2vulkan/build_support.rs", "\n");
        write("metal2vulkan/src/lib.rs", "pub fn translate() {}\n");
        write("metal2vulkan-runtime/Cargo.toml", "[package]\n");
        write("metal2vulkan-runtime/src/lib.rs", "pub fn submit() {}\n");
        write("metal2vulkan-devtools/Cargo.toml", "[package]\n");
        write("metal2vulkan-devtools/src/lib.rs", "pub fn assemble() {}\n");
        for input in &inputs {
            write(
                &format!("metal2vulkan-validation/{input}"),
                "// classifier\n",
            );
        }
        write("metal2vulkan-validation/src/ab.rs", "fn compare() {}\n");
        let validation = root.join("metal2vulkan-validation");
        let key = |version: &str| analyzer_abi(version, &product(root), &classifier(&validation));
        let current = key(analyzer_version!());

        write(
            "metal2vulkan-validation/src/ab.rs",
            "fn compare() { /* edited */ }\n",
        );
        write("metal2vulkan-validation/src/budgets.rs", "fn report() {}\n");
        assert_eq!(
            key(analyzer_version!()),
            current,
            "other validation edits are not inputs"
        );
        assert_ne!(
            key("structural-triage-v0"),
            current,
            "a version bump invalidates"
        );
        for input in &inputs {
            write(
                &format!("metal2vulkan-validation/{input}"),
                "// classifier, edited\n",
            );
            assert_ne!(
                key(analyzer_version!()),
                current,
                "an edit to {input} invalidates"
            );
            write(
                &format!("metal2vulkan-validation/{input}"),
                "// classifier\n",
            );
        }
        assert_eq!(key(analyzer_version!()), current);
        write(
            "metal2vulkan-devtools/src/lib.rs",
            "pub fn assemble() { /* edited */ }\n",
        );
        assert_ne!(
            key(analyzer_version!()),
            current,
            "a dev-tools edit invalidates"
        );
        write("metal2vulkan-devtools/src/lib.rs", "pub fn assemble() {}\n");
        assert_eq!(key(analyzer_version!()), current);
        write(
            "metal2vulkan-runtime/src/lib.rs",
            "pub fn submit() { /* edited */ }\n",
        );
        assert_ne!(
            key(analyzer_version!()),
            current,
            "a runtime edit invalidates"
        );
        write("metal2vulkan-runtime/src/lib.rs", "pub fn submit() {}\n");
        assert_eq!(key(analyzer_version!()), current);
        write(
            "metal2vulkan/src/lib.rs",
            "pub fn translate() { /* edited */ }\n",
        );
        assert_ne!(
            key(analyzer_version!()),
            current,
            "a product edit invalidates"
        );
    }

    #[test]
    fn cache_reuses_only_the_current_analyzer_contract() {
        let scratch = ScratchDir::new("triage-cache").unwrap();
        let index = scratch.path().join("index.sqlite");
        let connection = Connection::open(&index).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE sources (air_sha256 TEXT PRIMARY KEY, stage TEXT NOT NULL);\n\
                 INSERT INTO sources VALUES ('air', 'Kernel');",
            )
            .unwrap();
        drop(connection);
        let result = classify(&source("define void @main() { ret void }")).unwrap();
        write_cached(&index, [("air", &result)]).unwrap();
        let hashes = vec!["air".to_string()];
        assert_eq!(
            read_cached(&index, &hashes).unwrap().get("air"),
            Some(&result)
        );

        let connection = Connection::open(&index).unwrap();
        connection
            .execute(
                "UPDATE triage_analysis SET analyzer_abi='obsolete' WHERE air_sha256='air'",
                [],
            )
            .unwrap();
        assert!(read_cached(&index, &hashes).unwrap().is_empty());
    }

    #[test]
    fn cached_air_inventory_reclassifies_without_source_rows() {
        let scratch = ScratchDir::new("triage-cache-air-reclassify").unwrap();
        let index = scratch.path().join("index.sqlite");
        let connection = Connection::open(&index).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE sources (air_sha256 TEXT PRIMARY KEY, stage TEXT NOT NULL);\n\
                 INSERT INTO sources VALUES ('air', 'Kernel');",
            )
            .unwrap();
        drop(connection);
        let symbol =
            "air.simdgroup_matrix_16x16x16_multiply_accumulate.f.f.v8f32.v8f16.v8f16.v8f32";
        let mut stale = classify(&source("define void @main() { ret void }")).unwrap();
        stale.air_calls.insert(symbol.into(), 3);
        stale.unrecognized_air_intrinsics.insert(symbol.into(), 3);
        write_cached(&index, [("air", &stale)]).unwrap();

        assert_eq!(reclassify_cached_air_intrinsics(&index).unwrap(), 1);
        let current = read_cached(&index, &["air".into()]).unwrap();
        assert!(current["air"].unrecognized_air_intrinsics.is_empty());
        assert_eq!(current["air"].air_calls[symbol], 3);
    }

    #[test]
    fn analyzer_upgrade_reopens_every_row_when_the_contract_affects_every_stage() {
        let scratch = ScratchDir::new("triage-cache-upgrade-scope").unwrap();
        let index = scratch.path().join("index.sqlite");
        let connection = Connection::open(&index).unwrap();
        connection
            .execute_batch(
                "CREATE TABLE sources (air_sha256 TEXT PRIMARY KEY, stage TEXT NOT NULL);\n\
                 INSERT INTO sources VALUES ('affected', 'Kernel'), ('supported', 'Vertex'), ('unrelated', 'Kernel'), ('fragment', 'Fragment');",
            )
            .unwrap();
        drop(connection);
        let mut affected = classify(&source("define void @main() { ret void }")).unwrap();
        affected.air_calls.insert(
            "air.intersect.intersection_function_buffer.triangle_data".into(),
            1,
        );
        let supported_marker = classify(&source(
            "define void @main() { ret void }\n!0 = !{i32 0, !\"air.visible_function_table\"}",
        ))
        .unwrap();
        let unrelated = classify(&source("define void @main() { ret void }")).unwrap();
        let fragment = unrelated.clone();
        write_cached(
            &index,
            [
                ("affected", &affected),
                ("supported", &supported_marker),
                ("unrelated", &unrelated),
                ("fragment", &fragment),
            ],
        )
        .unwrap();
        let connection = Connection::open(&index).unwrap();
        connection
            .execute(
                "UPDATE triage_analysis SET analyzer_abi=?1",
                ["structural-triage-v13"],
            )
            .unwrap();
        drop(connection);

        let hashes = ["affected", "supported", "unrelated", "fragment"]
            .map(str::to_string)
            .to_vec();
        assert!(read_cached(&index, &hashes).unwrap().is_empty());
        assert_eq!(
            select_uncached(&index, 10).unwrap(),
            ["affected", "fragment", "supported", "unrelated"]
        );

        let connection = Connection::open(&index).unwrap();
        let versions = ["affected", "supported", "unrelated", "fragment"]
            .into_iter()
            .map(|hash| {
                connection
                    .query_row(
                        "SELECT analyzer_abi FROM triage_analysis WHERE air_sha256=?1",
                        [hash],
                        |row| row.get::<_, String>(0),
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert_eq!(versions, ["structural-triage-v13"; 4]);
    }

    #[test]
    fn framebuffer_fetch_uses_the_shared_supported_reflection_contract() {
        let mut row = source(
            r#"define <4 x float> @frag(<4 x float> %color) { ret <4 x float> %color }
!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{!4}
!4 = !{i32 0, !"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
"#,
        );
        row.stage = "Fragment".into();
        row.entry = "frag".into();
        // Every executor runs it (the engine's candidate through the input attachment its plan
        // reads), so the row carries no requirement.
        assert_eq!(classify(&row).unwrap().tooling_requirements, []);
    }

    /// A kernel `name` whose argument buffer (Metal buffer 0, constant space) is `{render_pipeline_state
    /// render; compute_pipeline_state compute; uint scale;}` and whose `ulong` output (Metal buffer
    /// 1) is `scale`: it reads no pipeline state.
    fn pipeline_state_kernel(name: &str) -> String {
        format!(
            r#"%ab = type {{ ptr addrspace(1), ptr addrspace(1), i32 }}

define void @{name}(ptr addrspace(2) %ab, ptr addrspace(1) %out) {{
  %ps = getelementptr inbounds %ab, ptr addrspace(2) %ab, i64 0, i32 2
  %s = load i32, ptr addrspace(2) %ps, align 4
  %z = zext i32 %s to i64
  store i64 %z, ptr addrspace(1) %out, align 8
  ret void
}}

!air.kernel = !{{!0}}
!air.version = !{{!9}}
!air.language_version = !{{!10}}
!0 = !{{ptr @{name}, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !8}}
!3 = !{{i32 0, !"air.indirect_buffer", !"air.buffer_size", i32 24, !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 2, !"air.struct_type_info", !4, !"air.arg_type_size", i32 24, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"Pipelines", !"air.arg_name", !"ab"}}
!4 = !{{i32 0, i32 8, i32 0, !"render_pipeline_state", !"render", !"air.indirect_argument", !5, i32 8, i32 8, i32 0, !"compute_pipeline_state", !"compute", !"air.indirect_argument", !6, i32 16, i32 4, i32 0, !"uint", !"scale", !"air.indirect_argument", !7}}
!5 = !{{i32 0, !"air.render_pipeline_state", !"air.location_index", i32 0, i32 1, !"air.arg_type_name", !"render_pipeline_state", !"air.arg_name", !"render"}}
!6 = !{{i32 1, !"air.compute_pipeline_state", !"air.location_index", i32 1, i32 1, !"air.arg_type_name", !"compute_pipeline_state", !"air.arg_name", !"compute"}}
!7 = !{{i32 2, !"air.indirect_constant", !"air.location_index", i32 2, i32 1, !"air.arg_type_name", !"uint", !"air.arg_name", !"scale"}}
!8 = !{{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"ulong", !"air.arg_name", !"out"}}
!9 = !{{i32 2, i32 8, i32 0}}
!10 = !{{!"Metal", i32 4, i32 0, i32 0}}
"#
        )
    }

    /// Every executor runs an argument buffer's pipeline-state members (the engine's
    /// candidate as an opaque id in the owner, which no translated pipeline reads), so
    /// the row carries no requirement and stays described by its descriptive fact, which a row
    /// without the family does not have.
    #[test]
    fn a_pipeline_state_row_is_described_not_required() {
        let row = source(&pipeline_state_kernel("main"));
        let classified = classify(&row).unwrap();
        assert_eq!(classified.tooling_requirements, []);
        assert!(classified.pipeline_state_members);
        let mut plain = source(include_str!(
            "../fixtures/public/fragment_constant_color.ll"
        ));
        let (stage, entry) = crate::air::stage_entry_from_ll(&plain.air_ll).expect("an entry");
        plain.stage = stage.into();
        plain.entry = entry;
        assert!(reflect(&plain).is_some());
        assert!(!classify(&plain).unwrap().pipeline_state_members);
    }

    /// Every executor runs a fragment function's explicit imageblock (the new
    /// engine's candidate in the storage image its plan names), so its row carries no
    /// requirement and stays described by its descriptive fact, which a row without the family
    /// does not have.
    #[test]
    fn a_fragment_imageblock_row_is_described_not_required() {
        let mut row = source(include_str!(
            "../fixtures/public/fragment_custom_imageblock.ll"
        ));
        let (stage, entry) = crate::air::stage_entry_from_ll(&row.air_ll).expect("an entry");
        row.stage = stage.into();
        row.entry = entry;
        let classified = classify(&row).unwrap();
        assert_eq!(classified.tooling_requirements, []);
        assert!(classified.fragment_imageblock);
        let mut plain = source(include_str!(
            "../fixtures/public/fragment_constant_color.ll"
        ));
        let (stage, entry) = crate::air::stage_entry_from_ll(&plain.air_ll).expect("an entry");
        plain.stage = stage.into();
        plain.entry = entry;
        assert!(!classify(&plain).unwrap().fragment_imageblock);
    }

    /// An object or mesh row is described by its draw: its stage, its payload, and the stages a row
    /// of it needs that nothing supplies, as the requirement it carries says (the object function
    /// draws with the generated launch observer of its payload, the mesh function with a generated
    /// object function that writes its payload); any other row has no mesh draw.
    #[test]
    fn a_mesh_row_is_described_by_the_stages_it_needs() {
        use crate::case::Stage;
        use crate::executor_contract::mesh::MeshDrawFacts;
        let facts = |entry: &str| {
            let classified = classify(&entry_of("engine/mesh_draws.ll", entry)).unwrap();
            let unresolved = classified
                .tooling_requirements
                .contains(&ToolingRequirement::MeshCompanion);
            let draw = classified.mesh_draw.expect("a mesh draw");
            assert_eq!(unresolved, !draw.missing.is_empty(), "{entry}");
            draw
        };
        assert_eq!(
            facts("mesh_draws_object"),
            MeshDrawFacts {
                stage: Stage::Object,
                payload: true,
                payload_size: Some(32),
                missing: vec![],
            }
        );
        assert_eq!(
            facts("mesh_draws_mesh"),
            MeshDrawFacts {
                stage: Stage::Mesh,
                payload: true,
                payload_size: Some(32),
                missing: vec![],
            }
        );
        assert_eq!(
            facts("mesh_draws_grid"),
            MeshDrawFacts {
                stage: Stage::Mesh,
                payload: false,
                payload_size: None,
                missing: vec![],
            }
        );
        let fragment = classify(&entry_of("engine/mesh_draws.ll", "mesh_draws_fragment")).unwrap();
        assert_eq!(fragment.mesh_draw, None);
    }

    /// A vertex function whose only effect is the indexed draw it encodes is observed by executing
    /// that draw, as a kernel's commands are: its row carries no requirement, the encoded commands
    /// an unobserved effect.
    #[test]
    fn a_vertex_row_whose_only_effect_is_its_replayed_commands_is_resolved() {
        for (module, entry) in [
            ("engine/icb_vertex_encode_indexed.ll", "ive_encode"),
            ("engine/icb_vertex_encode_textured.ll", "ivt_encode"),
        ] {
            let classified = classify(&entry_of(module, entry)).unwrap();
            assert_eq!(classified.tooling_requirements, [], "{entry}");
            assert!(!classified.unobserved_calls.is_empty(), "{entry}");
            assert!(
                classified
                    .unobserved_calls
                    .values()
                    .all(|effect| *effect == UnobservedEffect::IndirectCommandContents),
                "{entry}: {:?}",
                classified.unobserved_calls
            );
        }
    }

    /// A tile function (a kernel's implicit imageblock over render targets) carries
    /// no requirement and stays described by its descriptive fact, which a kernel without the
    /// family does not have.
    #[test]
    fn a_tile_function_row_is_described_not_required() {
        let mut row = source(include_str!(
            "../fixtures/public/kernel_implicit_imageblock_three_widths.ll"
        ));
        row.stage = "Kernel".into();
        row.entry = "kernel_implicit_imageblock_three_widths".into();
        let classified = classify(&row).unwrap();
        assert_eq!(classified.tooling_requirements, []);
        assert!(classified.tile_function);
        let mut plain = source(include_str!("../fixtures/public/kernel_imageblock_word.ll"));
        let (stage, entry) = crate::air::stage_entry_from_ll(&plain.air_ll).expect("an entry");
        plain.stage = stage.into();
        plain.entry = entry;
        assert!(!classify(&plain).unwrap().tile_function);
    }

    #[test]
    fn observation_requirements_classify_declared_varying_type_names() {
        let mut supported = source(
            r#"define <4 x float> @frag(<4 x i1> %payload) {
entry:
  ret <4 x float> zeroinitializer
}
!air.fragment = !{!0}
!0 = !{ptr @frag, !1, !3}
!1 = !{!2}
!2 = !{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}
!3 = !{!4}
!4 = !{i32 0, !"air.fragment_input", !"user(payload)", !"air.arg_type_name", !"bool4", !"air.arg_name", !"not.valid"}
"#,
        );
        supported.stage = "Fragment".into();
        supported.entry = "frag".into();
        assert!(classify(&supported)
            .unwrap()
            .tooling_requirements
            .is_empty());

        let unsupported = SourceRow {
            air_ll: supported.air_ll.replace("!\"bool4\"", "!\"double2\""),
            ..supported
        };
        assert_eq!(
            classify(&unsupported).unwrap().tooling_requirements,
            [ToolingRequirement::FragmentVaryingObservationType]
        );
    }

    #[test]
    fn vertex_without_position_is_an_explicit_observation_gap() {
        let mut row = source(
            r#"define <4 x float> @vertex() {
entry:
  ret <4 x float> zeroinitializer
}
!air.vertex = !{!0}
!0 = !{ptr @vertex, !1, !2}
!1 = !{!3}
!2 = !{}
!3 = !{!"air.vertex_output", !"user(payload)", !"air.arg_type_name", !"float4", !"air.arg_name", !"payload"}
"#,
        );
        row.stage = "Vertex".into();
        row.entry = "vertex".into();
        assert_eq!(
            classify(&row).unwrap().tooling_requirements,
            [ToolingRequirement::VertexSideEffectObservation]
        );
    }

    /// The row of entry `entry` of the committed public module `name`: the module's text, standing
    /// for that entry.
    fn entry_of(name: &str, entry: &str) -> SourceRow {
        let air_ll = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("fixtures/public")
                .join(name),
        )
        .unwrap();
        let declared = crate::air::module_entries(&air_ll)
            .unwrap()
            .into_iter()
            .find(|declared| declared.name == entry)
            .unwrap_or_else(|| panic!("{name} declares no entry {entry}"));
        SourceRow {
            air_sha256: crate::hash::sha256_bytes(air_ll.as_bytes()),
            stage: declared.stage.into(),
            entry: entry.into(),
            air_ll,
            blob_b64: None,
            lib_sha256s: vec!["owned-synthetic".into()],
            label: format!("public/{name}"),
        }
    }

    /// The facts of the whole module text, as a row of the module was once charged: what each
    /// entry's facts must not be.
    fn module_wide(row: &SourceRow) -> StructuralTriage {
        classify_inner(row, false)
    }

    /// The measured misattributions come out per entry. `sampling_completion`'s pixel samplers are
    /// `sample_pixel_zero`'s and `sample_pixel_nearest`'s, not every entry's:
    /// `sample_mixed_samplers` (one repeat-linear sampler) has none. The cycles of `fragment_rog`
    /// and `fragment_discard_msl22` are `rog_loop`'s and `discard_loop`'s alone.
    #[test]
    fn the_d119_4_misattributions_are_charged_to_the_entries_that_reach_them() {
        let module = "engine/sampling_completion.ll";
        let pixel_samplers = |row: &SourceRow| {
            let (_, interface) = reflect(&entry_row(row).unwrap()).unwrap();
            interface
                .bindings
                .iter()
                .filter_map(|binding| binding.static_sampler)
                .filter(|state| state.coordinates == crate::air::reflect::SamplerCoordinates::Pixel)
                .count()
        };
        assert_eq!(
            pixel_samplers(&entry_of(module, "sample_mixed_samplers")),
            0
        );
        assert_eq!(pixel_samplers(&entry_of(module, "sample_pixel_zero")), 1);

        for (module, looping) in [
            ("engine/fragment_rog.ll", "rog_loop"),
            ("engine/fragment_discard_msl22.ll", "discard_loop"),
        ] {
            let text = std::fs::read_to_string(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("fixtures/public")
                    .join(module),
            )
            .unwrap();
            let entries = crate::air::module_entries(&text).unwrap();
            assert!(entries.len() > 1, "{module}");
            for entry in entries {
                let row = entry_of(module, &entry.name);
                assert!(module_wide(&row).has_cfg_cycle, "{module}");
                assert_eq!(
                    classify(&row).unwrap().has_cfg_cycle,
                    entry.name == looping,
                    "{module} {}",
                    entry.name
                );
            }
        }
    }

    /// Each entry of a module with several is charged with exactly what it reaches: a helper two
    /// entries share (its loop) is charged to both, the static initializer to every entry, a
    /// function reached only through a global's initializer to the entry that reads the global, and
    /// argument metadata to its own entry.
    #[test]
    fn each_entry_of_a_module_is_charged_with_what_it_reaches() {
        let module = crate::air::projection::tests::THREE_KERNELS;
        let row = |entry: &str| SourceRow {
            entry: entry.into(),
            ..source(module)
        };
        let calls = |facts: &StructuralTriage| facts.air_calls.keys().cloned().collect::<Vec<_>>();
        let marker = |facts: &StructuralTriage, kind: &str| facts.resource_markers[kind];
        let whole = module_wide(&row("a"));
        assert!(whole.has_cfg_cycle);
        let (a, b, c) = (
            classify(&row("a")).unwrap(),
            classify(&row("b")).unwrap(),
            classify(&row("c")).unwrap(),
        );
        assert!(a.has_cfg_cycle && b.has_cfg_cycle && !c.has_cfg_cycle);
        assert_eq!(calls(&a), ["air.a_only", "air.init_call"]);
        assert_eq!(calls(&b), ["air.init_call"]);
        assert_eq!(calls(&c), ["air.c_only", "air.init_call", "air.table_only"]);
        assert_eq!(
            calls(&whole),
            [
                "air.a_only",
                "air.c_only",
                "air.init_call",
                "air.table_only"
            ]
        );
        assert_eq!(
            a.unrecognized_air_intrinsics.keys().collect::<Vec<_>>(),
            ["air.a_only", "air.init_call"]
        );
        assert_eq!(
            (marker(&a, "air.buffer"), marker(&a, "air.texture")),
            (1, 0)
        );
        assert_eq!(
            (marker(&b, "air.buffer"), marker(&b, "air.texture")),
            (0, 1)
        );
        assert_eq!(
            (marker(&c, "air.buffer"), marker(&c, "air.render_target")),
            (0, 1)
        );
        assert_eq!(marker(&whole, "air.buffer"), 1);
        assert_eq!(marker(&whole, "air.texture"), 1);
    }

    /// The product reflects each entry's projection as it reflects the entry in its module, but
    /// for the static samplers, which the old product's `inspect` lists for the whole module: the
    /// projection keeps only the entry's.
    #[test]
    fn the_product_reflects_each_projection_as_its_entry_in_the_module() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures/public");
        let without_static_samplers = |mut interface: crate::air::reflect::ShaderInterface| {
            interface
                .bindings
                .retain(|binding| binding.static_sampler.is_none());
            interface
        };
        let static_samplers = |interface: &crate::air::reflect::ShaderInterface| {
            interface
                .bindings
                .iter()
                .filter(|binding| binding.static_sampler.is_some())
                .count()
        };
        let mut reflected = 0;
        let mut fewer_samplers = 0;
        for file in std::fs::read_dir(root.join("engine")).unwrap() {
            let name = file.unwrap().file_name().into_string().unwrap();
            if !name.ends_with(".ll") || name.ends_with(".air.ll") {
                continue;
            }
            let module = format!("engine/{name}");
            let text = std::fs::read_to_string(root.join(&module)).unwrap();
            if crate::air::entry_count_from_ll(&text) < 2 {
                continue;
            }
            for entry in crate::air::module_entries(&text).unwrap() {
                let row = entry_of(&module, &entry.name);
                let projected = entry_row(&row).unwrap();
                assert!(matches!(projected, Cow::Owned(_)), "{module}");
                let (whole, own) = (reflect(&row), reflect(&projected));
                assert_eq!(whole.is_some(), own.is_some(), "{module} {}", entry.name);
                let (Some((_, whole)), Some((_, own))) = (whole, own) else {
                    continue;
                };
                reflected += 1;
                assert!(static_samplers(&own) <= static_samplers(&whole));
                fewer_samplers += usize::from(static_samplers(&own) < static_samplers(&whole));
                assert_eq!(
                    without_static_samplers(own),
                    without_static_samplers(whole),
                    "{module} {}",
                    entry.name
                );
            }
        }
        assert!(reflected > 100, "{reflected} entries reflected");
        assert!(fewer_samplers > 0);
    }
}
