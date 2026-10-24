//! AIR modules that are not shader entry points.
//!
//! Visible/intersection function implementations are commonly packaged as separate AIR blobs with
//! no shader-stage entry metadata. They are dependencies of authored entry cases, not independently
//! executable queue rows, so harvest retains them separately instead of dropping them.
//!
//! A case may also link a public module: owned MSL `[[visible]]` functions under
//! `validation/fixtures/public/linked/`, which `scripts/regen-air-fixtures --dir linked` compiles.
//! Such a module is found by its `.ll` text's hash before any private shard, retains no blob, and
//! each consumer builds its bytes from its text ([`LibraryModuleRow::bitcode`] for the engine
//! and the checker; the Metal oracle assembles its typed `.air.ll` with `metal-as`).

use crate::case::{
    AuthoredCase, FunctionTableEntry, FunctionTableResource, IntersectionFunctionSignature,
    IntersectionFunctionTableEntry, LinkedFunctionResource,
};
use crate::hash::sha256_bytes;
use crate::jsonl::to_sorted_json_string;
use crate::source::{shard_index_for_hash, shard_name};
use base64::Engine as _;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LibraryModuleRow {
    pub module_sha256: String,
    pub air_ll: String,
    pub blob_b64: String,
    pub lib_sha256s: Vec<String>,
    pub label: String,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LibraryModuleMergeStats {
    pub affected_shards: usize,
    pub inserted: usize,
    pub merged_memberships: usize,
    pub duplicates: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LibraryModuleIndexSyncStats {
    pub shards_scanned: usize,
    pub bytes_scanned: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LibraryModuleShardStamp {
    size: u64,
    modified_ns: u64,
}

/// Fully checked function-table dependencies for one authored entry point.
///
/// Case checking resolves these once by content hash and verifies that every implementation
/// defines the exact authored symbol. Executors consume this object directly, so they cannot
/// accidentally use a different module lookup or silently omit an authored table entry.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ResolvedLinkedFunctions {
    pub references: Vec<ResolvedFunctionReference>,
    pub visible: Vec<ResolvedFunctionTable>,
    pub intersection: Vec<ResolvedIntersectionFunctionTable>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedFunctionReference {
    pub function: String,
    pub module: LibraryModuleRow,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedFunctionTable {
    pub binding: u32,
    pub size: u32,
    pub entries: Vec<ResolvedFunctionEntry>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedFunctionEntry {
    pub index: u32,
    pub function: String,
    pub module: LibraryModuleRow,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedIntersectionFunctionTable {
    pub location: ResolvedIntersectionFunctionTableLocation,
    pub size: u32,
    pub entries: Vec<ResolvedIntersectionFunctionEntry>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResolvedIntersectionFunctionTableLocation {
    Direct {
        binding: u32,
    },
    ArgumentBuffer {
        buffer_binding: u32,
        field_offset: u32,
    },
}

impl ResolvedIntersectionFunctionTableLocation {
    pub fn buffer_binding(self) -> u32 {
        match self {
            Self::Direct { binding } => binding,
            Self::ArgumentBuffer { buffer_binding, .. } => buffer_binding,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolvedIntersectionFunctionEntry {
    Linked(ResolvedFunctionEntry),
    OpaqueTriangle {
        index: u32,
        signature: Vec<IntersectionFunctionSignature>,
    },
}

impl ResolvedIntersectionFunctionEntry {
    pub fn index(&self) -> u32 {
        match self {
            Self::Linked(entry) => entry.index,
            Self::OpaqueTriangle { index, .. } => *index,
        }
    }
}

impl ResolvedLinkedFunctions {
    pub fn is_empty(&self) -> bool {
        self.references.is_empty() && self.visible.is_empty() && self.intersection.is_empty()
    }

    pub fn all_dependencies(&self) -> impl Iterator<Item = (&str, &LibraryModuleRow)> {
        self.references
            .iter()
            .map(|reference| (reference.function.as_str(), &reference.module))
            .chain(self.visible.iter().flat_map(|table| {
                table
                    .entries
                    .iter()
                    .map(|entry| (entry.function.as_str(), &entry.module))
            }))
            .chain(self.intersection.iter().flat_map(|table| {
                table.entries.iter().filter_map(|entry| match entry {
                    ResolvedIntersectionFunctionEntry::Linked(entry) => {
                        Some((entry.function.as_str(), &entry.module))
                    }
                    ResolvedIntersectionFunctionEntry::OpaqueTriangle { .. } => None,
                })
            }))
    }

    /// The libraries a pipeline stage links functions from, as both executors link them (Metal's
    /// `MTLLinkedFunctions.functions`, the engine's `LinkedFunctions.functions`): each distinct
    /// module of [`Self::all_dependencies`] once, in first appearance, with each distinct function
    /// linked from it, in first appearance. A function named twice (a reference and a table entry,
    /// or two table slots) is one linked function.
    pub fn libraries(&self) -> Vec<LinkedLibraryFunctions<'_>> {
        let mut libraries = Vec::<LinkedLibraryFunctions<'_>>::new();
        for (function, module) in self.all_dependencies() {
            let at = match libraries
                .iter()
                .position(|library| library.module.module_sha256 == module.module_sha256)
            {
                Some(at) => at,
                None => {
                    libraries.push(LinkedLibraryFunctions {
                        module,
                        functions: Vec::new(),
                    });
                    libraries.len() - 1
                }
            };
            let functions = &mut libraries[at].functions;
            if !functions.contains(&function) {
                functions.push(function);
            }
        }
        libraries
    }
}

/// One library a pipeline stage links functions from ([`ResolvedLinkedFunctions::libraries`]):
/// the module, loaded once, and the name of each function linked from it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinkedLibraryFunctions<'a> {
    pub module: &'a LibraryModuleRow,
    pub functions: Vec<&'a str>,
}

/// Resolve every authored table entry through its one hash-derived module shard.
///
/// All errors are accumulated so an author can repair a manifest in one pass. Repeated references
/// to the same module are loaded once, while the returned tables retain the authored ordering that
/// literal validation already proved canonical.
pub fn resolve_linked_functions(
    root: &Path,
    case: &AuthoredCase,
) -> Result<ResolvedLinkedFunctions, Vec<String>> {
    let mut modules = HashMap::<String, Result<Option<LibraryModuleRow>, String>>::new();
    let mut defined_functions = HashMap::<String, Vec<String>>::new();
    let mut errors = Vec::new();
    let references = resolve_reference_kind(
        root,
        &case.visible_function_references,
        &mut modules,
        &mut defined_functions,
        &mut errors,
    );
    let visible = resolve_table_kind(
        root,
        "visible-function",
        &case.visible_function_tables,
        &mut modules,
        &mut defined_functions,
        &mut errors,
    );
    let authored_intersection = case
        .intersection_function_tables
        .iter()
        .map(|table| {
            (
                ResolvedIntersectionFunctionTableLocation::Direct {
                    binding: table.binding,
                },
                table.size,
                table.entries.as_slice(),
            )
        })
        .chain(
            case.argument_buffer_intersection_function_tables
                .iter()
                .map(|table| {
                    (
                        ResolvedIntersectionFunctionTableLocation::ArgumentBuffer {
                            buffer_binding: table.buffer_binding,
                            field_offset: table.field_offset,
                        },
                        table.size,
                        table.entries.as_slice(),
                    )
                }),
        )
        .collect::<Vec<_>>();
    let linked_intersection = authored_intersection
        .iter()
        .map(|(location, size, entries)| FunctionTableResource {
            binding: location.buffer_binding(),
            size: *size,
            entries: entries
                .iter()
                .filter_map(|entry| match entry {
                    IntersectionFunctionTableEntry::Linked {
                        index,
                        module_sha256,
                        function,
                    } => Some(FunctionTableEntry {
                        index: *index,
                        module_sha256: module_sha256.clone(),
                        function: function.clone(),
                    }),
                    IntersectionFunctionTableEntry::OpaqueTriangle { .. } => None,
                })
                .collect(),
        })
        .collect::<Vec<_>>();
    let resolved_linked = resolve_table_kind(
        root,
        "intersection-function",
        &linked_intersection,
        &mut modules,
        &mut defined_functions,
        &mut errors,
    );
    let intersection = authored_intersection
        .iter()
        .zip(resolved_linked)
        .map(|((location, size, entries), resolved)| {
            let mut linked = resolved
                .entries
                .into_iter()
                .map(|entry| (entry.index, entry))
                .collect::<HashMap<_, _>>();
            ResolvedIntersectionFunctionTable {
                location: *location,
                size: *size,
                entries: entries
                    .iter()
                    .filter_map(|entry| match entry {
                        IntersectionFunctionTableEntry::Linked { index, .. } => linked
                            .remove(index)
                            .map(ResolvedIntersectionFunctionEntry::Linked),
                        IntersectionFunctionTableEntry::OpaqueTriangle { index, signature } => {
                            Some(ResolvedIntersectionFunctionEntry::OpaqueTriangle {
                                index: *index,
                                signature: signature.clone(),
                            })
                        }
                    })
                    .collect(),
            }
        })
        .collect::<Vec<_>>();
    let mut linked_names = HashMap::<&str, &str>::new();
    for (function, module_sha256) in references
        .iter()
        .map(|reference| {
            (
                reference.function.as_str(),
                reference.module.module_sha256.as_str(),
            )
        })
        .chain(
            visible
                .iter()
                .flat_map(|table| &table.entries)
                .chain(intersection.iter().flat_map(|table| {
                    table.entries.iter().filter_map(|entry| match entry {
                        ResolvedIntersectionFunctionEntry::Linked(entry) => Some(entry),
                        ResolvedIntersectionFunctionEntry::OpaqueTriangle { .. } => None,
                    })
                }))
                .map(|entry| (entry.function.as_str(), entry.module.module_sha256.as_str())),
        )
    {
        if let Some(previous_module) = linked_names.insert(function, module_sha256) {
            if previous_module != module_sha256 {
                errors.push(format!(
                    "linked function {:?} is defined by both module {} and module {}; Metal requires linked function names to be unique",
                    function, previous_module, module_sha256
                ));
            }
        }
    }
    if errors.is_empty() {
        Ok(ResolvedLinkedFunctions {
            references,
            visible,
            intersection,
        })
    } else {
        Err(errors)
    }
}

fn resolve_reference_kind(
    root: &Path,
    references: &[LinkedFunctionResource],
    modules: &mut HashMap<String, Result<Option<LibraryModuleRow>, String>>,
    defined_functions: &mut HashMap<String, Vec<String>>,
    errors: &mut Vec<String>,
) -> Vec<ResolvedFunctionReference> {
    references
        .iter()
        .filter_map(|reference| {
            let module = modules
                .entry(reference.module_sha256.clone())
                .or_insert_with(|| find_library_module(root, &reference.module_sha256));
            let module = match module {
                Ok(Some(module)) => module,
                Ok(None) => {
                    errors.push(format!(
                        "visible function reference {:?} module {} is not harvested",
                        reference.function, reference.module_sha256
                    ));
                    return None;
                }
                Err(error) => {
                    errors.push(error.clone());
                    return None;
                }
            };
            let names = defined_functions
                .entry(module.module_sha256.clone())
                .or_insert_with(|| defined_function_names(&module.air_ll));
            if !names.contains(&reference.function) {
                errors.push(format!(
                    "visible function reference module {} does not define function {:?}",
                    reference.module_sha256, reference.function
                ));
                return None;
            }
            Some(ResolvedFunctionReference {
                function: reference.function.clone(),
                module: module.clone(),
            })
        })
        .collect()
}

fn resolve_table_kind(
    root: &Path,
    kind: &str,
    tables: &[FunctionTableResource],
    modules: &mut HashMap<String, Result<Option<LibraryModuleRow>, String>>,
    defined_functions: &mut HashMap<String, Vec<String>>,
    errors: &mut Vec<String>,
) -> Vec<ResolvedFunctionTable> {
    tables
        .iter()
        .map(|table| {
            let entries = table
                .entries
                .iter()
                .filter_map(|entry| {
                    let module = modules
                        .entry(entry.module_sha256.clone())
                        .or_insert_with(|| find_library_module(root, &entry.module_sha256));
                    let module = match module {
                        Ok(Some(module)) => module,
                        Ok(None) => {
                            errors.push(format!(
                                "{kind} table binding {} entry {} module {} is not harvested",
                                table.binding, entry.index, entry.module_sha256
                            ));
                            return None;
                        }
                        Err(error) => {
                            errors.push(error.clone());
                            return None;
                        }
                    };
                    let names = defined_functions
                        .entry(module.module_sha256.clone())
                        .or_insert_with(|| defined_function_names(&module.air_ll));
                    if !names.contains(&entry.function) {
                        errors.push(format!(
                            "{kind} table binding {} entry {} module {} does not define function {:?}",
                            table.binding, entry.index, entry.module_sha256, entry.function
                        ));
                        return None;
                    }
                    Some(ResolvedFunctionEntry {
                        index: entry.index,
                        function: entry.function.clone(),
                        module: module.clone(),
                    })
                })
                .collect();
            ResolvedFunctionTable {
                binding: table.binding,
                size: table.size,
                entries,
            }
        })
        .collect()
}

impl LibraryModuleRow {
    /// Whether the module is a public linked module ([`find_public_library_module`]): it retains
    /// no blob, and its bytes are built from its text.
    pub fn is_public(&self) -> bool {
        self.blob_b64.is_empty() && crate::source::public_source_path(&self.label).is_some()
    }

    /// The module's bitcode as the engine and the checker read it: the retained blob, or a
    /// public module's `.ll` text assembled with `llvm-as` (as a public source row's is).
    pub fn bitcode(&self) -> Result<Vec<u8>, String> {
        if self.is_public() {
            return crate::engine::RowSource::Text(std::borrow::Cow::Borrowed(&self.air_ll))
                .bitcode()
                .map_err(|error| format!("assemble public linked module {}: {error}", self.label));
        }
        base64::engine::general_purpose::STANDARD
            .decode(&self.blob_b64)
            .map_err(|error| {
                format!(
                    "decode library module {}'s retained bitcode: {error}",
                    self.module_sha256
                )
            })
    }

    pub fn validate(&self) -> Result<(), String> {
        let computed = sha256_bytes(self.air_ll.as_bytes());
        if self.module_sha256 != computed {
            return Err(format!(
                "library module {} hash mismatch: row={} computed={computed}",
                self.label, self.module_sha256
            ));
        }
        if !self
            .air_ll
            .lines()
            .any(|line| line.trim_start().starts_with("define "))
        {
            return Err(format!(
                "library module {} contains no function definition",
                self.label
            ));
        }
        let blob = base64::engine::general_purpose::STANDARD
            .decode(&self.blob_b64)
            .map_err(|error| format!("library module {} has invalid blob: {error}", self.label))?;
        if blob.is_empty() {
            return Err(format!("library module {} has an empty blob", self.label));
        }
        if self.lib_sha256s.is_empty() {
            return Err(format!(
                "library module {} has no parent library",
                self.label
            ));
        }
        let mut previous = None;
        for hash in &self.lib_sha256s {
            validate_hash("parent library", hash)?;
            if previous.is_some_and(|value: &String| value >= hash) {
                return Err(format!(
                    "library module {} parent libraries must be sorted and unique",
                    self.label
                ));
            }
            previous = Some(hash);
        }
        if self.label.trim().is_empty() {
            return Err("library module label must not be empty".into());
        }
        Ok(())
    }
}

/// The directory under `validation/fixtures/public` of the public linked modules. Not a public
/// source directory (`source::PUBLIC_SOURCE_DIRS`): its modules declare no entry point, so they are
/// no rows of their own.
pub const PUBLIC_LINKED_DIR: &str = "linked";

/// Each public linked module's text hash and file name (`<name>.ll`, never the typed
/// `<name>.air.ll`), read once per process: the fixtures are part of this crate's sources.
fn public_linked_modules() -> Result<&'static BTreeMap<String, String>, String> {
    static MODULES: std::sync::OnceLock<Result<BTreeMap<String, String>, String>> =
        std::sync::OnceLock::new();
    MODULES
        .get_or_init(|| {
            let directory = public_linked_dir();
            let entries = match fs::read_dir(&directory) {
                Ok(entries) => entries,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    return Ok(BTreeMap::new())
                }
                Err(error) => return Err(format!("read {}: {error}", directory.display())),
            };
            let mut modules = BTreeMap::new();
            for entry in entries {
                let path = entry
                    .map_err(|error| format!("read {}: {error}", directory.display()))?
                    .path();
                let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                    continue;
                };
                if !name.ends_with(".ll") || name.ends_with(".air.ll") {
                    continue;
                }
                let text =
                    fs::read(&path).map_err(|error| format!("read {}: {error}", path.display()))?;
                modules.insert(sha256_bytes(&text), name.to_string());
            }
            Ok(modules)
        })
        .as_ref()
        .map_err(Clone::clone)
}

fn public_linked_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures/public")
        .join(PUBLIC_LINKED_DIR)
}

/// The public linked module whose `.ll` text hashes to `module_sha256`, as a row labelled
/// `public/linked/<name>.ll` that retains no blob and belongs to no library.
pub fn find_public_library_module(module_sha256: &str) -> Result<Option<LibraryModuleRow>, String> {
    let Some(name) = public_linked_modules()?.get(module_sha256) else {
        return Ok(None);
    };
    let path = public_linked_dir().join(name);
    let air_ll =
        fs::read_to_string(&path).map_err(|error| format!("read {}: {error}", path.display()))?;
    if sha256_bytes(air_ll.as_bytes()) != module_sha256 {
        return Err(format!(
            "public linked module {} changed while this process ran",
            path.display()
        ));
    }
    Ok(Some(LibraryModuleRow {
        module_sha256: module_sha256.to_string(),
        air_ll,
        blob_b64: String::new(),
        lib_sha256s: Vec::new(),
        label: format!("public/{PUBLIC_LINKED_DIR}/{name}"),
    }))
}

/// Every public linked module, in label order.
pub fn public_library_modules() -> Result<Vec<LibraryModuleRow>, String> {
    let mut modules = Vec::new();
    for module_sha256 in public_linked_modules()?.keys() {
        modules.extend(find_public_library_module(module_sha256)?);
    }
    modules.sort_by(|a, b| a.label.cmp(&b.label));
    Ok(modules)
}

pub fn library_modules_dir(root: &Path) -> PathBuf {
    crate::source::private_data_dir(root).join("library-modules")
}

pub fn library_module_shard_path(root: &Path, shard: usize) -> PathBuf {
    library_modules_dir(root).join(shard_name(shard))
}

pub fn read_library_module_shard(path: &Path) -> Result<Vec<LibraryModuleRow>, String> {
    let mut rows = Vec::new();
    for_each_library_module_shard(path, |row| {
        rows.push(row);
        Ok(())
    })?;
    Ok(rows)
}

/// Stream fully validated dependency modules while retaining at most one decoded row at a time.
pub fn for_each_library_module_shard(
    path: &Path,
    mut consume: impl FnMut(LibraryModuleRow) -> Result<(), String>,
) -> Result<(), String> {
    let expected = crate::source::shard_index_from_path(path)?;
    let file = File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    for (line_index, line) in BufReader::new(file).lines().enumerate() {
        let line =
            line.map_err(|error| format!("read {}:{}: {error}", path.display(), line_index + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let row: LibraryModuleRow = serde_json::from_str(&line)
            .map_err(|error| format!("parse {}:{}: {error}", path.display(), line_index + 1))?;
        row.validate()
            .map_err(|error| format!("{}:{}: {error}", path.display(), line_index + 1))?;
        let actual = shard_index_for_hash(&row.module_sha256)?;
        if actual != expected {
            return Err(format!(
                "{}:{}: library module {} belongs in shard {}, not {}",
                path.display(),
                line_index + 1,
                row.module_sha256,
                actual,
                expected
            ));
        }
        consume(row)?;
    }
    Ok(())
}

/// Resolve one explicitly authored dependency by its content hash: a public linked module
/// ([`find_public_library_module`]) first, so a case over one checks without the private corpus,
/// then the private store.
///
/// The hash determines the only shard that may contain the module; no corpus-wide search is ever
/// needed for case checking or execution.
pub fn find_library_module(
    root: &Path,
    module_sha256: &str,
) -> Result<Option<LibraryModuleRow>, String> {
    if let Some(module) = find_public_library_module(module_sha256)? {
        return Ok(Some(module));
    }
    let shard = shard_index_for_hash(module_sha256)?;
    let path = library_module_shard_path(root, shard);
    if !path.is_file() {
        return Ok(None);
    }
    Ok(read_library_module_shard(&path)?
        .into_iter()
        .find(|row| row.module_sha256 == module_sha256))
}

/// Incrementally index retained dependency-module identities, library memberships, symbols, and
/// exact JSONL byte locations.
///
/// A missing index is an explicit one-time migration. After it is warm, unchanged module shards
/// are identified from their stamps and are not opened.
pub fn sync_library_module_index(
    root: &Path,
    index: &Path,
) -> Result<LibraryModuleIndexSyncStats, String> {
    let mut connection = Connection::open(index)
        .map_err(|error| format!("open index {}: {error}", index.display()))?;
    crate::index::keep_write_ahead_log(&connection, index)?;
    connection
        .execute_batch(
            "PRAGMA foreign_keys=ON;
             CREATE TABLE IF NOT EXISTS library_modules (
               module_sha256 TEXT PRIMARY KEY,
               shard INTEGER NOT NULL,
               source_offset INTEGER NOT NULL,
               source_length INTEGER NOT NULL
             );
             CREATE TABLE IF NOT EXISTS library_module_memberships (
               module_sha256 TEXT NOT NULL REFERENCES library_modules(module_sha256) ON DELETE CASCADE,
               lib_sha256 TEXT NOT NULL,
               PRIMARY KEY(module_sha256, lib_sha256)
             );
             CREATE TABLE IF NOT EXISTS library_module_symbols (
               module_sha256 TEXT NOT NULL REFERENCES library_modules(module_sha256) ON DELETE CASCADE,
               symbol TEXT NOT NULL,
               PRIMARY KEY(module_sha256, symbol)
             );
             CREATE TABLE IF NOT EXISTS indexed_library_module_shards (
               shard INTEGER PRIMARY KEY,
               size INTEGER NOT NULL,
               modified_ns INTEGER NOT NULL
             );
             CREATE INDEX IF NOT EXISTS library_memberships_by_library
               ON library_module_memberships(lib_sha256, module_sha256);
             CREATE INDEX IF NOT EXISTS library_symbols_by_symbol
               ON library_module_symbols(symbol, module_sha256);",
        )
        .map_err(|error| format!("upgrade library-module index schema: {error}"))?;

    let paths = library_module_shard_paths(root)?;
    let current_shards = paths
        .iter()
        .map(|path| crate::source::shard_index_from_path(path))
        .collect::<Result<std::collections::HashSet<_>, _>>()?;
    let transaction = connection
        .transaction()
        .map_err(|error| format!("begin library-module index sync: {error}"))?;
    let tracked_shards = {
        let mut statement = transaction
            .prepare("SELECT shard FROM indexed_library_module_shards")
            .map_err(|error| format!("prepare tracked library-module shards: {error}"))?;
        let shards = statement
            .query_map([], |row| row.get::<_, usize>(0))
            .map_err(|error| format!("query tracked library-module shards: {error}"))?
            .collect::<Result<std::collections::HashSet<_>, _>>()
            .map_err(|error| format!("read tracked library-module shards: {error}"))?;
        shards
    };
    for shard in tracked_shards.difference(&current_shards) {
        transaction
            .execute("DELETE FROM library_modules WHERE shard=?1", [shard])
            .map_err(|error| format!("remove library-module shard {shard}: {error}"))?;
        transaction
            .execute(
                "DELETE FROM indexed_library_module_shards WHERE shard=?1",
                [shard],
            )
            .map_err(|error| format!("remove library-module shard stamp {shard}: {error}"))?;
    }

    let mut stats = LibraryModuleIndexSyncStats::default();
    for path in paths {
        let shard = crate::source::shard_index_from_path(&path)?;
        let stamp = library_module_shard_stamp(&path)?;
        let tracked = transaction
            .query_row(
                "SELECT size, modified_ns FROM indexed_library_module_shards WHERE shard=?1",
                [shard],
                |row| {
                    Ok(LibraryModuleShardStamp {
                        size: row.get(0)?,
                        modified_ns: row.get(1)?,
                    })
                },
            )
            .optional()
            .map_err(|error| format!("read library-module shard {shard} stamp: {error}"))?;
        if tracked == Some(stamp) {
            continue;
        }
        transaction
            .execute("DELETE FROM library_modules WHERE shard=?1", [shard])
            .map_err(|error| format!("replace library-module shard {shard}: {error}"))?;
        index_library_module_shard(&transaction, &path, shard)?;
        transaction
            .execute(
                "INSERT OR REPLACE INTO indexed_library_module_shards (shard, size, modified_ns)
                 VALUES (?1, ?2, ?3)",
                params![shard, stamp.size, stamp.modified_ns],
            )
            .map_err(|error| format!("record library-module shard {shard}: {error}"))?;
        stats.shards_scanned += 1;
        stats.bytes_scanned = stats
            .bytes_scanned
            .checked_add(stamp.size)
            .ok_or_else(|| "library-module shard byte count overflow".to_string())?;
    }
    transaction
        .commit()
        .map_err(|error| format!("commit library-module index sync: {error}"))?;
    Ok(stats)
}

/// The retained modules of the parent libraries `parent_library_sha256s` that define `symbol`,
/// in hash order, and the one module when exactly one does (its exact byte range read), from the
/// `(library, symbol)` index.
pub fn indexed_symbol_definers(
    root: &Path,
    index: &Path,
    symbol: &str,
    parent_library_sha256s: &[String],
) -> Result<(Vec<String>, Option<LibraryModuleRow>), String> {
    let connection = open_index_read_only(index)?;
    let hashes = indexed_modules_for_symbol(&connection, parent_library_sha256s, symbol)?;
    let module = match hashes.as_slice() {
        [hash] => Some(
            find_indexed_library_module(root, &connection, hash)?
                .ok_or_else(|| format!("indexed library module {hash} is unavailable"))?,
        ),
        _ => None,
    };
    Ok((hashes, module))
}

fn indexed_modules_for_symbol(
    connection: &Connection,
    libraries: &[String],
    symbol: &str,
) -> Result<Vec<String>, String> {
    let mut statement = connection
        .prepare(
            "SELECT m.module_sha256
             FROM library_module_memberships m
             JOIN library_module_symbols s ON s.module_sha256=m.module_sha256
             WHERE m.lib_sha256=?1 AND s.symbol=?2
             ORDER BY m.module_sha256",
        )
        .map_err(|error| format!("prepare linked-symbol lookup: {error}"))?;
    let mut hashes = BTreeSet::new();
    for library in libraries {
        let rows = statement
            .query_map(params![library, symbol], |row| row.get(0))
            .map_err(|error| format!("query linked symbol {symbol:?}: {error}"))?;
        for hash in rows {
            hashes.insert(hash.map_err(|error| format!("read linked symbol {symbol:?}: {error}"))?);
        }
    }
    Ok(hashes.into_iter().collect())
}

fn find_indexed_library_module(
    root: &Path,
    connection: &Connection,
    hash: &str,
) -> Result<Option<LibraryModuleRow>, String> {
    let location = connection
        .query_row(
            "SELECT shard, source_offset, source_length FROM library_modules WHERE module_sha256=?1",
            [hash],
            |row| Ok((row.get::<_, usize>(0)?, row.get::<_, u64>(1)?, row.get::<_, usize>(2)?)),
        )
        .optional()
        .map_err(|error| format!("query library module {hash}: {error}"))?;
    let Some((shard, offset, length)) = location else {
        return Ok(None);
    };
    let path = library_module_shard_path(root, shard);
    let mut file =
        File::open(&path).map_err(|error| format!("open {}: {error}", path.display()))?;
    read_located_module(&mut file, &path, hash, offset, length).map(Some)
}

/// The library module `hash` from its indexed byte range of the open shard `file` at `path`; the
/// row read must be the module the location names.
fn read_located_module(
    file: &mut File,
    path: &Path,
    hash: &str,
    offset: u64,
    length: usize,
) -> Result<LibraryModuleRow, String> {
    file.seek(SeekFrom::Start(offset))
        .map_err(|error| format!("seek {} to module {hash}: {error}", path.display()))?;
    let mut bytes = vec![0; length];
    file.read_exact(&mut bytes)
        .map_err(|error| format!("read module {hash} from {}: {error}", path.display()))?;
    let row: LibraryModuleRow = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse indexed library module {hash}: {error}"))?;
    if row.module_sha256 != hash {
        return Err(format!(
            "indexed library-module location for {hash} contains {}",
            row.module_sha256
        ));
    }
    Ok(row)
}

/// What a read of selected library modules through the index touched
/// ([`for_each_indexed_library_module`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IndexedLibraryModuleReadStats {
    pub rows: usize,
    pub shards_opened: usize,
    pub bytes_read: u64,
}

/// A read-only connection to the index. A worker's lookups read it while its lane's parent commits
/// results; the index's write-ahead log ([`crate::index::keep_write_ahead_log`]) lets them read its
/// last committed state without waiting on that writer.
fn open_index_read_only(index: &Path) -> Result<Connection, String> {
    Connection::open_with_flags(
        index,
        rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|error| format!("open index {}: {error}", index.display()))
}

/// Bring the library-module index up to date with the module shards on disk, then select up to
/// `limit` indexed library modules after `after`, in hash order (a resumable batch: pass the
/// previous batch's last hash as `after`). This is the bitcode-reader audit's `--library-modules`
/// selection.
///
/// The sync is [`sync_library_module_index`]: incremental, so it opens only a shard that is new
/// or whose size or modification time differs from what the index recorded (a copied corpus whose
/// stamps lost their nanoseconds is scanned once), and a warm index opens none. It returns what
/// the sync scanned with the selection.
pub fn select_current_library_modules(
    root: &Path,
    index: &Path,
    after: Option<&str>,
    limit: usize,
) -> Result<(LibraryModuleIndexSyncStats, Vec<String>), String> {
    let synced = sync_library_module_index(root, index)?;
    let hashes = select_indexed_library_modules(root, index, after, limit)?;
    Ok((synced, hashes))
}

/// Select up to `limit` indexed library modules after `after`, in hash order, without writing the
/// index.
///
/// The selection is complete only over a current library-module index, so every module shard on
/// disk must carry the size and modification time the index recorded for it, and every recorded
/// shard must still exist. Otherwise the selection fails naming the shard instead of silently
/// leaving its modules out. [`select_current_library_modules`] syncs the index first, so this
/// refuses there only when a shard changed in between. Only the shards' metadata is read here,
/// never their contents.
fn select_indexed_library_modules(
    root: &Path,
    index: &Path,
    after: Option<&str>,
    limit: usize,
) -> Result<Vec<String>, String> {
    let connection = open_index_read_only(index)?;
    let mut tracked = HashMap::new();
    {
        let mut statement = connection
            .prepare("SELECT shard, size, modified_ns FROM indexed_library_module_shards")
            .map_err(|error| format!("prepare library-module shard stamps: {error}"))?;
        let rows = statement
            .query_map([], |row| {
                Ok((
                    row.get::<_, usize>(0)?,
                    LibraryModuleShardStamp {
                        size: row.get(1)?,
                        modified_ns: row.get(2)?,
                    },
                ))
            })
            .map_err(|error| format!("query library-module shard stamps: {error}"))?;
        for row in rows {
            let (shard, stamp) =
                row.map_err(|error| format!("read library-module shard stamp: {error}"))?;
            tracked.insert(shard, stamp);
        }
    }
    let mut current = BTreeSet::new();
    for path in library_module_shard_paths(root)? {
        let shard = crate::source::shard_index_from_path(&path)?;
        if tracked.get(&shard) != Some(&library_module_shard_stamp(&path)?) {
            return Err(format!(
                "library-module shard {} changed since the index recorded it; sync the library-module index",
                path.display()
            ));
        }
        current.insert(shard);
    }
    if let Some(shard) = tracked.keys().find(|shard| !current.contains(shard)) {
        return Err(format!(
            "the index records library-module shard {shard}, which no longer exists; sync the library-module index"
        ));
    }
    let mut statement = connection
        .prepare(
            "SELECT module_sha256 FROM library_modules WHERE (?1 IS NULL OR module_sha256 > ?1) \
             ORDER BY module_sha256 LIMIT ?2",
        )
        .map_err(|error| format!("prepare library-module selection: {error}"))?;
    let limit = i64::try_from(limit).unwrap_or(i64::MAX);
    let hashes = statement
        .query_map(params![after, limit], |row| row.get::<_, String>(0))
        .map_err(|error| format!("query library-module selection: {error}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("read library-module selection: {error}"))?;
    Ok(hashes)
}

/// Stream the selected library modules from their indexed byte ranges, in `hashes` order, without
/// writing the index: only the shards that hold a selected module are opened, and only the
/// selected rows' bytes are read. Each row read must be the module its location names.
pub fn for_each_indexed_library_module(
    root: &Path,
    index: &Path,
    hashes: &[String],
    mut consume: impl FnMut(LibraryModuleRow) -> Result<(), String>,
) -> Result<IndexedLibraryModuleReadStats, String> {
    let connection = open_index_read_only(index)?;
    let mut statement = connection
        .prepare(
            "SELECT shard, source_offset, source_length FROM library_modules \
             WHERE module_sha256=?1",
        )
        .map_err(|error| format!("prepare library-module locations: {error}"))?;
    let mut files = HashMap::<usize, File>::new();
    let mut stats = IndexedLibraryModuleReadStats::default();
    for hash in hashes {
        let (shard, offset, length) = statement
            .query_row([hash], |row| {
                Ok((
                    row.get::<_, usize>(0)?,
                    row.get::<_, u64>(1)?,
                    row.get::<_, usize>(2)?,
                ))
            })
            .optional()
            .map_err(|error| format!("query library module {hash}: {error}"))?
            .ok_or_else(|| format!("library module {hash} is not indexed"))?;
        let path = library_module_shard_path(root, shard);
        let file = match files.entry(shard) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => entry.insert(
                File::open(&path).map_err(|error| format!("open {}: {error}", path.display()))?,
            ),
        };
        let row = read_located_module(file, &path, hash, offset, length)?;
        stats.rows += 1;
        stats.bytes_read = stats
            .bytes_read
            .checked_add(length as u64)
            .ok_or_else(|| "library-module byte count overflow".to_string())?;
        consume(row)?;
    }
    stats.shards_opened = files.len();
    Ok(stats)
}

fn library_module_shard_paths(root: &Path) -> Result<Vec<PathBuf>, String> {
    let directory = library_modules_dir(root);
    if !directory.is_dir() {
        return Ok(Vec::new());
    }
    let mut paths = fs::read_dir(&directory)
        .map_err(|error| format!("read {}: {error}", directory.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "jsonl")
        })
        .collect::<Vec<_>>();
    paths.sort();
    Ok(paths)
}

fn library_module_shard_stamp(path: &Path) -> Result<LibraryModuleShardStamp, String> {
    let metadata =
        fs::metadata(path).map_err(|error| format!("read metadata {}: {error}", path.display()))?;
    let modified_ns = metadata
        .modified()
        .map_err(|error| format!("read modification time {}: {error}", path.display()))?
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("invalid modification time {}: {error}", path.display()))?
        .as_nanos()
        .try_into()
        .map_err(|_| format!("modification time overflow for {}", path.display()))?;
    Ok(LibraryModuleShardStamp {
        size: metadata.len(),
        modified_ns,
    })
}

fn index_library_module_shard(
    transaction: &rusqlite::Transaction<'_>,
    path: &Path,
    shard: usize,
) -> Result<(), String> {
    let mut reader = BufReader::new(
        File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?,
    );
    let mut offset = 0u64;
    let mut line = Vec::new();
    loop {
        line.clear();
        let length = reader
            .read_until(b'\n', &mut line)
            .map_err(|error| format!("read {}: {error}", path.display()))?;
        if length == 0 {
            break;
        }
        let row_bytes = line.strip_suffix(b"\n").unwrap_or(&line);
        if row_bytes.is_empty() {
            offset += length as u64;
            continue;
        }
        let row: LibraryModuleRow = serde_json::from_slice(row_bytes)
            .map_err(|error| format!("parse {} at byte {offset}: {error}", path.display()))?;
        row.validate()
            .map_err(|error| format!("{} at byte {offset}: {error}", path.display()))?;
        let actual = shard_index_for_hash(&row.module_sha256)?;
        if actual != shard {
            return Err(format!(
                "library module {} belongs in shard {actual}, not {shard}",
                row.module_sha256
            ));
        }
        transaction
            .execute(
                "INSERT INTO library_modules (module_sha256, shard, source_offset, source_length)
                 VALUES (?1, ?2, ?3, ?4)",
                params![row.module_sha256, shard, offset, row_bytes.len()],
            )
            .map_err(|error| format!("index library module {}: {error}", row.module_sha256))?;
        for library in &row.lib_sha256s {
            transaction
                .execute(
                    "INSERT INTO library_module_memberships (module_sha256, lib_sha256)
                     VALUES (?1, ?2)",
                    params![row.module_sha256, library],
                )
                .map_err(|error| format!("index library membership: {error}"))?;
        }
        for symbol in defined_function_names(&row.air_ll) {
            transaction
                .execute(
                    "INSERT INTO library_module_symbols (module_sha256, symbol) VALUES (?1, ?2)",
                    params![row.module_sha256, symbol],
                )
                .map_err(|error| format!("index library symbol: {error}"))?;
        }
        offset += length as u64;
    }
    Ok(())
}

pub fn defined_function_names(ll: &str) -> Vec<String> {
    ll.lines()
        .filter_map(|line| {
            let line = line.trim_start();
            if !line.starts_with("define ") {
                return None;
            }
            let at = line.find('@')?;
            llvm_symbol(&line[at + 1..])
        })
        .collect()
}

fn llvm_symbol(text: &str) -> Option<String> {
    if let Some(quoted) = text.strip_prefix('"') {
        let mut escaped = false;
        let end = quoted.char_indices().find_map(|(index, ch)| {
            if escaped {
                escaped = false;
                None
            } else if ch == '\\' {
                escaped = true;
                None
            } else if ch == '"' {
                Some(index)
            } else {
                None
            }
        })?;
        Some(quoted[..end].to_string())
    } else {
        let end = text.find('(')?;
        let name = text[..end].trim();
        (!name.is_empty()).then(|| name.to_string())
    }
}

pub fn merge_library_module_shards(
    root: &Path,
    rows: impl IntoIterator<Item = LibraryModuleRow>,
) -> Result<LibraryModuleMergeStats, String> {
    let mut additions = BTreeMap::<usize, Vec<LibraryModuleRow>>::new();
    for row in rows {
        row.validate()?;
        additions
            .entry(shard_index_for_hash(&row.module_sha256)?)
            .or_default()
            .push(row);
    }
    if additions.is_empty() {
        return Ok(LibraryModuleMergeStats::default());
    }
    fs::create_dir_all(library_modules_dir(root))
        .map_err(|error| format!("create library-module shards: {error}"))?;
    remove_stale_temporaries(root)?;
    let mut stats = LibraryModuleMergeStats {
        affected_shards: additions.len(),
        ..LibraryModuleMergeStats::default()
    };
    for (shard, additions) in additions {
        let path = library_module_shard_path(root, shard);
        let existing = if path.is_file() {
            read_library_module_shard(&path)?
        } else {
            Vec::new()
        };
        let mut merged = existing
            .into_iter()
            .map(|row| (row.module_sha256.clone(), row))
            .collect::<BTreeMap<_, _>>();
        for addition in additions {
            match merged.get_mut(&addition.module_sha256) {
                None => {
                    stats.inserted += 1;
                    merged.insert(addition.module_sha256.clone(), addition);
                }
                Some(existing) => {
                    let before = existing.lib_sha256s.len();
                    existing.lib_sha256s.extend(addition.lib_sha256s);
                    existing.lib_sha256s.sort();
                    existing.lib_sha256s.dedup();
                    stats.merged_memberships += existing.lib_sha256s.len() - before;
                    if addition.blob_b64 < existing.blob_b64 {
                        existing.blob_b64 = addition.blob_b64;
                    }
                    if existing.lib_sha256s.len() == before {
                        stats.duplicates += 1;
                    }
                }
            }
        }
        write_library_module_bucket(root, shard, merged.into_values().collect())?;
    }
    Ok(stats)
}

fn write_library_module_bucket(
    root: &Path,
    shard: usize,
    mut rows: Vec<LibraryModuleRow>,
) -> Result<(), String> {
    rows.sort_by(|left, right| left.module_sha256.cmp(&right.module_sha256));
    let path = library_module_shard_path(root, shard);
    let temporary = library_modules_dir(root).join(format!(
        ".{}.{}.tmp",
        shard_name(shard),
        std::process::id()
    ));
    let result = (|| {
        let mut file = File::create(&temporary)
            .map_err(|error| format!("create {}: {error}", temporary.display()))?;
        for row in rows {
            let line = to_sorted_json_string(&row)
                .map_err(|error| format!("serialize {}: {error}", row.module_sha256))?;
            writeln!(file, "{line}")
                .map_err(|error| format!("write {}: {error}", temporary.display()))?;
        }
        file.sync_all()
            .map_err(|error| format!("fsync {}: {error}", temporary.display()))?;
        fs::rename(&temporary, &path).map_err(|error| {
            format!(
                "rename {} to {}: {error}",
                temporary.display(),
                path.display()
            )
        })?;
        File::open(library_modules_dir(root))
            .and_then(|directory| directory.sync_all())
            .map_err(|error| {
                format!(
                    "fsync directory {}: {error}",
                    library_modules_dir(root).display()
                )
            })
    })();
    if let Err(error) = result {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    Ok(())
}

fn remove_stale_temporaries(root: &Path) -> Result<(), String> {
    let directory = library_modules_dir(root);
    for path in fs::read_dir(&directory)
        .map_err(|error| format!("read {}: {error}", directory.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
    {
        let stale = path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.starts_with(".shard_") && name.ends_with(".tmp"));
        if stale {
            fs::remove_file(&path)
                .map_err(|error| format!("remove stale {}: {error}", path.display()))?;
        }
    }
    Ok(())
}

fn validate_hash(label: &str, hash: &str) -> Result<(), String> {
    if hash.len() != 64
        || !hash.bytes().all(|byte| byte.is_ascii_hexdigit())
        || hash.bytes().any(|byte| byte.is_ascii_uppercase())
    {
        Err(format!("{label} must be a lowercase SHA-256"))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ScratchDir;

    fn row(ll: &str, library: &str) -> LibraryModuleRow {
        LibraryModuleRow {
            module_sha256: sha256_bytes(ll.as_bytes()),
            air_ll: ll.into(),
            blob_b64: base64::engine::general_purpose::STANDARD.encode(b"owned bitcode"),
            lib_sha256s: vec![library.into()],
            label: "local/library-module.ll".into(),
        }
    }

    #[test]
    fn merge_is_shard_local_and_preserves_all_library_memberships() {
        let scratch = ScratchDir::new("library-module-merge").unwrap();
        let first = row("define void @visible() { ret void }", &"11".repeat(32));
        let first_shard = shard_index_for_hash(&first.module_sha256).unwrap();
        let mut unrelated = first.clone();
        for nonce in 0..u32::MAX {
            unrelated = row(
                &format!("define void @other() {{ ret void }}\n; {nonce}"),
                &"22".repeat(32),
            );
            if shard_index_for_hash(&unrelated.module_sha256).unwrap() != first_shard {
                break;
            }
        }
        merge_library_module_shards(scratch.path(), [first.clone(), unrelated.clone()]).unwrap();
        let unrelated_path = library_module_shard_path(
            scratch.path(),
            shard_index_for_hash(&unrelated.module_sha256).unwrap(),
        );
        let unrelated_bytes = fs::read(&unrelated_path).unwrap();
        fs::write(&unrelated_path, vec![b'x'; unrelated_bytes.len()]).unwrap();

        let same_module_other_library = row(&first.air_ll, &"33".repeat(32));
        let stats =
            merge_library_module_shards(scratch.path(), [same_module_other_library]).unwrap();
        assert_eq!(stats.affected_shards, 1);
        assert_eq!(stats.merged_memberships, 1);
        assert_eq!(
            fs::read(&unrelated_path).unwrap(),
            vec![b'x'; unrelated_bytes.len()]
        );
        assert_eq!(
            read_library_module_shard(&library_module_shard_path(scratch.path(), first_shard))
                .unwrap()[0]
                .lib_sha256s,
            vec!["11".repeat(32), "33".repeat(32)]
        );
    }

    /// A symbol resolves through every parent library of the entry, its dependencies' included.
    #[test]
    fn indexed_symbols_resolve_across_all_entry_library_memberships() {
        let scratch = ScratchDir::new("indexed-visible-reference").unwrap();
        let entry_library = "11".repeat(32);
        let dependency_library = "22".repeat(32);
        let dependency = row("define void @linked() { ret void }\n", &dependency_library);
        merge_library_module_shards(scratch.path(), [dependency.clone()]).unwrap();
        let index = scratch.path().join("index.sqlite");
        Connection::open(&index).unwrap();

        let cold = sync_library_module_index(scratch.path(), &index).unwrap();
        assert_eq!(cold.shards_scanned, 1);
        let warm = sync_library_module_index(scratch.path(), &index).unwrap();
        assert_eq!(warm, LibraryModuleIndexSyncStats::default());

        let (definers, module) = indexed_symbol_definers(
            scratch.path(),
            &index,
            "linked",
            &[entry_library, dependency_library],
        )
        .unwrap();
        assert_eq!(definers, std::slice::from_ref(&dependency.module_sha256));
        assert_eq!(module.unwrap().module_sha256, dependency.module_sha256);
    }

    /// The read-only library-module selection is resumable in hash order, reads only the selected
    /// modules' shards and bytes, and refuses an index that no longer matches the shards on disk
    /// instead of selecting fewer modules.
    #[test]
    fn indexed_library_modules_select_in_hash_order_and_read_only_their_shards() {
        let scratch = ScratchDir::new("indexed-library-module-selection").unwrap();
        let library = "11".repeat(32);
        let modules: Vec<LibraryModuleRow> = (0..6)
            .map(|n| row(&format!("define void @f{n}() {{ ret void }}\n"), &library))
            .collect();
        merge_library_module_shards(scratch.path(), modules.clone()).unwrap();
        let index = scratch.path().join("index.sqlite");
        Connection::open(&index).unwrap();
        sync_library_module_index(scratch.path(), &index).unwrap();
        let mut hashes: Vec<String> = modules.iter().map(|m| m.module_sha256.clone()).collect();
        hashes.sort();

        let all = select_indexed_library_modules(scratch.path(), &index, None, 100).unwrap();
        assert_eq!(all, hashes);
        let first = select_indexed_library_modules(scratch.path(), &index, None, 2).unwrap();
        assert_eq!(first, hashes[..2]);
        let rest =
            select_indexed_library_modules(scratch.path(), &index, first.last().map(|s| &**s), 100)
                .unwrap();
        assert_eq!(rest, hashes[2..]);

        // Reading one module opens its shard alone and reads only its row.
        let mut read = Vec::new();
        let stats = for_each_indexed_library_module(scratch.path(), &index, &hashes[..1], |m| {
            read.push(m);
            Ok(())
        })
        .unwrap();
        let module = modules
            .iter()
            .find(|m| m.module_sha256 == hashes[0])
            .unwrap();
        assert_eq!(read, std::slice::from_ref(module));
        assert_eq!(stats.rows, 1);
        assert_eq!(stats.shards_opened, 1);
        assert_eq!(
            stats.bytes_read,
            to_sorted_json_string(module).unwrap().len() as u64
        );
        // A hash the index does not hold is an error, not a silently shorter batch.
        let missing =
            for_each_indexed_library_module(scratch.path(), &index, &["00".repeat(32)], |_| Ok(()))
                .unwrap_err();
        assert!(missing.contains("not indexed"), "{missing}");

        // A module added after the index was synced makes the selection refuse, naming its shard.
        let late = row("define void @late() { ret void }\n", &library);
        merge_library_module_shards(scratch.path(), [late.clone()]).unwrap();
        let stale = select_indexed_library_modules(scratch.path(), &index, None, 100).unwrap_err();
        let shard = shard_index_for_hash(&late.module_sha256).unwrap();
        assert!(stale.contains(&shard_name(shard)), "{stale}");
        sync_library_module_index(scratch.path(), &index).unwrap();
        assert_eq!(
            select_indexed_library_modules(scratch.path(), &index, None, 100)
                .unwrap()
                .len(),
            7
        );
        // So does a recorded shard that is gone.
        fs::remove_file(library_module_shard_path(scratch.path(), shard)).unwrap();
        let gone = select_indexed_library_modules(scratch.path(), &index, None, 100).unwrap_err();
        assert!(gone.contains("no longer exists"), "{gone}");
    }

    /// The bitcode-reader audit's `--library-modules` selection syncs the index first, so it
    /// selects every module even where nothing else synced it: a shard whose stamp lost its
    /// nanoseconds in a copy (`rsync -t`, `tar`) with the same bytes, a module added since, and a
    /// shard that is gone. The read-only selection refuses each of them; the syncing one scans only
    /// the changed shards, and a warm one scans none.
    #[test]
    fn the_library_module_selection_syncs_a_stale_index_first() {
        let scratch = ScratchDir::new("current-library-module-selection").unwrap();
        let library = "11".repeat(32);
        let modules: Vec<LibraryModuleRow> = (0..6)
            .map(|n| row(&format!("define void @f{n}() {{ ret void }}\n"), &library))
            .collect();
        merge_library_module_shards(scratch.path(), modules.clone()).unwrap();
        let index = scratch.path().join("index.sqlite");
        Connection::open(&index).unwrap();
        let (cold, _) = select_current_library_modules(scratch.path(), &index, None, 100).unwrap();
        assert!(cold.shards_scanned > 0);
        let mut hashes: Vec<String> = modules.iter().map(|m| m.module_sha256.clone()).collect();
        hashes.sort();
        let (warm, all) =
            select_current_library_modules(scratch.path(), &index, None, 100).unwrap();
        assert_eq!(warm, LibraryModuleIndexSyncStats::default());
        assert_eq!(all, hashes);

        // The same bytes under a whole-second modification time, as a copy keeps it.
        let copied =
            library_module_shard_path(scratch.path(), shard_index_for_hash(&hashes[0]).unwrap());
        let second = UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_000);
        File::options()
            .write(true)
            .open(&copied)
            .unwrap()
            .set_modified(second)
            .unwrap();
        let stale = select_indexed_library_modules(scratch.path(), &index, None, 100).unwrap_err();
        assert!(
            stale.contains("changed since the index recorded it"),
            "{stale}"
        );
        let (resynced, selected) =
            select_current_library_modules(scratch.path(), &index, None, 100).unwrap();
        assert_eq!(resynced.shards_scanned, 1);
        assert_eq!(resynced.bytes_scanned, fs::metadata(&copied).unwrap().len());
        assert_eq!(selected, hashes);

        // A module added since the last sync, and then its shard gone.
        let late = row("define void @late() { ret void }\n", &library);
        merge_library_module_shards(scratch.path(), [late.clone()]).unwrap();
        let unsynced =
            select_indexed_library_modules(scratch.path(), &index, None, 100).unwrap_err();
        assert!(
            unsynced.contains("changed since the index recorded it"),
            "{unsynced}"
        );
        let (_, with_late) =
            select_current_library_modules(scratch.path(), &index, None, 100).unwrap();
        assert_eq!(with_late.len(), 7);
        assert!(with_late.contains(&late.module_sha256));
        let late_shard = shard_index_for_hash(&late.module_sha256).unwrap();
        let late_path = library_module_shard_path(scratch.path(), late_shard);
        let kept = read_library_module_shard(&late_path)
            .unwrap()
            .into_iter()
            .filter(|m| m.module_sha256 != late.module_sha256)
            .count();
        fs::remove_file(&late_path).unwrap();
        let gone = select_indexed_library_modules(scratch.path(), &index, None, 100).unwrap_err();
        assert!(gone.contains("no longer exists"), "{gone}");
        let (_, without_late) =
            select_current_library_modules(scratch.path(), &index, None, 100).unwrap();
        assert_eq!(without_late.len(), 6 - kept);
        assert!(!without_late.contains(&late.module_sha256));
    }

    /// A symbol two retained modules define resolves to neither: authoring decides.
    #[test]
    fn indexed_symbols_leave_ambiguous_definitions_for_authoring() {
        let scratch = ScratchDir::new("ambiguous-visible-reference").unwrap();
        let library = "11".repeat(32);
        let first = row("define void @linked() { ret void }\n; first\n", &library);
        let second = row("define void @linked() { ret void }\n; second\n", &library);
        merge_library_module_shards(scratch.path(), [first, second]).unwrap();
        let index = scratch.path().join("index.sqlite");
        Connection::open(&index).unwrap();
        sync_library_module_index(scratch.path(), &index).unwrap();

        let (definers, module) = indexed_symbol_definers(
            scratch.path(),
            &index,
            "linked",
            std::slice::from_ref(&library),
        )
        .unwrap();
        assert_eq!(definers.len(), 2);
        assert_eq!(module, None);
    }

    /// Workers read the index while a writer holds a transaction open: a lane's parent commits
    /// results every few rows while its workers' linkage lookups read the index, and with a
    /// rollback journal a commit locked them out until their busy timeout failed them. Here eight
    /// readers run every worker lookup while the writer holds an exclusive transaction that the
    /// readers outlast; under a rollback journal each would wait on it and fail. The index starts
    /// as an older binary built it, in rollback-journal mode, and the library-module sync switches
    /// it; a plain connection, as an older binary opens it, still reads and writes.
    #[test]
    fn workers_read_the_index_while_a_writer_holds_a_transaction() {
        let scratch = ScratchDir::new("indexed-visible-reference-writer").unwrap();
        let library = "11".repeat(32);
        let dependency = row("define void @linked() { ret void }\n", &library);
        merge_library_module_shards(scratch.path(), [dependency.clone()]).unwrap();
        let index = scratch.path().join("index.sqlite");
        let journal_mode = |connection: &Connection| -> String {
            connection
                .query_row("PRAGMA journal_mode", [], |row| row.get(0))
                .unwrap()
        };
        let older = Connection::open(&index).unwrap();
        older.execute_batch("PRAGMA journal_mode=DELETE").unwrap();
        assert_eq!(journal_mode(&older), "delete");
        drop(older);
        sync_library_module_index(scratch.path(), &index).unwrap();

        let writer = Connection::open(&index).unwrap();
        assert_eq!(journal_mode(&writer), "wal");
        writer
            .execute_batch(
                "BEGIN EXCLUSIVE;
                 INSERT INTO indexed_library_module_shards (shard, size, modified_ns)
                 VALUES (999, 1, 1);",
            )
            .unwrap();
        let readers = (0..8)
            .map(|_| {
                let (root, index) = (scratch.path().to_path_buf(), index.clone());
                let (library, module) = (library.clone(), dependency.module_sha256.clone());
                std::thread::spawn(move || {
                    for _ in 0..4 {
                        let (_, resolved) = indexed_symbol_definers(
                            &root,
                            &index,
                            "linked",
                            std::slice::from_ref(&library),
                        )
                        .unwrap();
                        assert_eq!(resolved.unwrap().module_sha256, module);
                        let mut read = Vec::new();
                        for_each_indexed_library_module(
                            &root,
                            &index,
                            std::slice::from_ref(&module),
                            |row| {
                                read.push(row.module_sha256);
                                Ok(())
                            },
                        )
                        .unwrap();
                        assert_eq!(read, std::slice::from_ref(&module));
                    }
                })
            })
            .collect::<Vec<_>>();
        for reader in readers {
            reader.join().unwrap();
        }
        writer.execute_batch("COMMIT").unwrap();
        drop(writer);

        let plain = Connection::open(&index).unwrap();
        let shards: i64 = plain
            .query_row(
                "SELECT count(*) FROM indexed_library_module_shards WHERE shard=999",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(shards, 1);
        plain
            .execute(
                "DELETE FROM indexed_library_module_shards WHERE shard=999",
                [],
            )
            .unwrap();
        assert_eq!(journal_mode(&plain), "wal");
    }

    #[test]
    fn module_requires_real_function_body_and_exact_identity() {
        let mut module = row("declare void @visible()", &"11".repeat(32));
        assert!(module
            .validate()
            .unwrap_err()
            .contains("no function definition"));
        module.air_ll = "define void @visible() { ret void }".into();
        assert!(module.validate().unwrap_err().contains("hash mismatch"));
    }

    #[test]
    fn function_names_cover_plain_and_quoted_llvm_symbols() {
        assert_eq!(
            defined_function_names(
                "define void @plain() { ret void }\ndefine float @\"quoted name\"(float %x) { ret float %x }\ndeclare void @ignored()"
            ),
            ["plain", "quoted name"]
        );
    }

    #[test]
    fn direct_reference_resolves_an_explicit_cross_library_module() {
        let scratch = ScratchDir::new("direct-linked-function").unwrap();
        let module = row(
            "define i32 @linked(i32 %value) { ret i32 %value }",
            &"33".repeat(32),
        );
        merge_library_module_shards(scratch.path(), [module.clone()]).unwrap();
        let mut modules = HashMap::new();
        let mut names = HashMap::new();
        let mut errors = Vec::new();
        let resolved = resolve_reference_kind(
            scratch.path(),
            &[LinkedFunctionResource {
                module_sha256: module.module_sha256.clone(),
                function: "linked".into(),
            }],
            &mut modules,
            &mut names,
            &mut errors,
        );
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].module, module);
    }

    /// Both executors link by one grouping: each module once, in first appearance, with each
    /// function linked from it once, whether a reference or a table slot names it.
    #[test]
    fn a_stage_links_each_module_once_with_each_function_once() {
        let first = row("define i32 @a(i32 %v) { ret i32 %v }", &"33".repeat(32));
        let second = row("define i32 @b(i32 %v) { ret i32 %v }", &"44".repeat(32));
        let reference = |function: &str, module: &LibraryModuleRow| ResolvedFunctionReference {
            function: function.into(),
            module: module.clone(),
        };
        let entry = |index: u32, function: &str, module: &LibraryModuleRow| ResolvedFunctionEntry {
            index,
            function: function.into(),
            module: module.clone(),
        };
        let linked = ResolvedLinkedFunctions {
            references: vec![
                reference("b", &second),
                reference("a", &first),
                reference("c", &second),
            ],
            visible: vec![ResolvedFunctionTable {
                binding: 1,
                size: 3,
                entries: vec![
                    entry(0, "a", &first),
                    entry(1, "b", &second),
                    entry(2, "d", &first),
                ],
            }],
            intersection: vec![],
        };
        let libraries = linked
            .libraries()
            .into_iter()
            .map(|library| (library.module.module_sha256.clone(), library.functions))
            .collect::<Vec<_>>();
        assert_eq!(
            libraries,
            [
                (second.module_sha256.clone(), vec!["b", "c"]),
                (first.module_sha256.clone(), vec!["a", "d"]),
            ]
        );
        assert!(ResolvedLinkedFunctions::default().libraries().is_empty());
    }

    #[test]
    fn shard_count_matches_source_store_contract() {
        assert_eq!(crate::source::SHARD_COUNT, 64);
    }

    /// The public linked modules resolve by their `.ll` text's hash with no private store: each
    /// labelled by its fixture, retaining no blob and no parent library, defining its owned
    /// functions, with the typed `.air.ll` the Metal oracle assembles beside it, and a direct
    /// reference resolves to one as to a retained module. Its bitcode for the engine and the
    /// checker is its text assembled with `llvm-as`.
    #[test]
    fn public_linked_modules_resolve_by_hash_without_a_private_store() {
        let modules = public_library_modules().unwrap();
        assert_eq!(
            modules
                .iter()
                .map(|module| (
                    module.label.as_str(),
                    defined_function_names(&module.air_ll)
                ))
                .collect::<Vec<_>>(),
            [
                (
                    "public/linked/mps_postfix_primary_4xf_to_last_buffer.ll",
                    vec!["postfixPrimary_4xf".to_string()]
                ),
                (
                    "public/linked/mps_prefix_secondary_tertiary_f_from_buffers.ll",
                    vec![
                        "prefixSecondary_f".to_string(),
                        "prefixTertiary_f".to_string()
                    ]
                ),
            ]
        );
        let scratch = ScratchDir::new("public-linked-module").unwrap();
        for module in &modules {
            assert!(module.is_public(), "{}", module.label);
            assert!(module.blob_b64.is_empty() && module.lib_sha256s.is_empty());
            assert_eq!(module.module_sha256, sha256_bytes(module.air_ll.as_bytes()));
            // No private store holds it: the checkout supplies it first.
            assert!(!library_modules_dir(scratch.path()).exists());
            assert_eq!(
                find_library_module(scratch.path(), &module.module_sha256).unwrap(),
                Some(module.clone())
            );
            let (typed, is_typed) = crate::source::public_metal_input(&module.label).unwrap();
            assert!(is_typed && typed.is_file(), "{}", typed.display());
            match metal2vulkan_devtools::llvm_assemble(
                &crate::source::public_source_path(&module.label)
                    .map(|path| {
                        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                            .join("../..")
                            .join(path)
                    })
                    .unwrap(),
            ) {
                Ok(bitcode) => assert_eq!(module.bitcode().unwrap(), bitcode),
                Err(metal2vulkan_devtools::AssembleError::Missing(why)) => {
                    eprintln!("SKIPPED the bitcode comparison: {why}")
                }
                Err(metal2vulkan_devtools::AssembleError::Rejected(why)) => {
                    panic!("llvm-as rejected {}: {why}", module.label)
                }
            }
        }
        assert_eq!(find_public_library_module(&"00".repeat(32)).unwrap(), None);
        // A retained module is never public, whatever it holds.
        assert!(!row("define void @f() { ret void }", &"11".repeat(32)).is_public());

        let secondary = &modules[1];
        let mut resolved_modules = HashMap::new();
        let mut names = HashMap::new();
        let mut errors = Vec::new();
        let resolved = resolve_reference_kind(
            scratch.path(),
            &[
                LinkedFunctionResource {
                    module_sha256: secondary.module_sha256.clone(),
                    function: "prefixTertiary_f".into(),
                },
                LinkedFunctionResource {
                    module_sha256: secondary.module_sha256.clone(),
                    function: "postfixPrimary_4xf".into(),
                },
            ],
            &mut resolved_modules,
            &mut names,
            &mut errors,
        );
        assert_eq!(resolved.len(), 1);
        assert_eq!(&resolved[0].module, secondary);
        assert_eq!(
            errors,
            [format!(
                "visible function reference module {} does not define function \
                 \"postfixPrimary_4xf\"",
                secondary.module_sha256
            )]
        );
    }
}
