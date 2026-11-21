# Authored validation corpus

Canonical state is plain, deterministically ordered JSONL. The generated `.index.sqlite` is
gitignored and may be deleted at any time.

```text
crates/metal2vulkan-validation/corpus/
  local/sources/shard_000.jsonl ... shard_063.jsonl  # private, gitignored
  local/library-modules/shard_000.jsonl ...          # non-entry AIR dependencies, private
  cases/shard_000.jsonl ...                         # authored, no AIR bodies
  observations/metal/shard_000.jsonl ...
  observations/vulkan/shard_000.jsonl ...
  observations/parity/<scenario>.jsonl              # Metal's records of the parity scenarios
  authored-compile.tsv                              # each case's compile outcome per profile
  .index.sqlite                                     # generated, gitignored
```

`observations/parity/sequences.jsonl` holds one line per Metal environment recorded on the Apple
M3 Max. Under macOS 27.0.1, the reference, every seed ran and matches the model. Under macOS 26.7
seeds 32, 51 and 62 are Metal-side gaps: Metal's driver crashed on an indirect dispatch in each,
and that line records the crash in place of the seed's records. The sequences lane names a gap
seed of the line it compares with in its output.

`authored-compile.tsv` is the authored compile census's expectation file: one line
per committed case, its outcome on each profile of its header, `compiled` or the executor's
refusal label. Only `corpus-triage --audit authored-compile` writes it: each train's gate adds new
cases and venues' profiles and drops removed cases (`--bless-additions`), and a change that
moves a case's outcome commits the file blessed with it (`--bless`; docs/VALIDATION.md, "Compile
every committed authored case").

Committed cases and observations may reference private AIR. A checkout with no private
store at all (no `local/`, and no `--private-corpus`) indexes only the rows of its public sources:
those of private AIR are expected and unavailable there, and `corpus-index` reports how many it left
out (`scope=public-only`). A checkout with a private store indexes every row, and a missing source
shard is an error.

All paths with the same shard number cover the same bucket. The bucket is the first six bits of
lowercase `air_sha256`. This is a permanent storage contract: do not change the bucket count or
mapping to rebalance the corpus.

The 64-bucket choice is based on a 2026-08-10 measurement of 43,656 private source rows. The
measured 32-, 64-, and 128-bucket layouts averaged 1,364.25, 682.12, and 341.06 rows per bucket,
respectively; their largest buckets held 1,451, 740, and 401 rows. Sixty-four was the smallest
layout with an average below 1,000 rows while retaining reasonably large sequential JSONL shards.

## Sources

Private source rows contain the SHA-256 of the exact sanitized text, stable AIR stage/entry
metadata, sanitized `air_ll`, optional `blob_b64`, the sorted `lib_sha256s` array of parent-library hashes, and a
non-semantic label.
Harvest output is sorted by AIR hash and byte-identical for the same extracted inputs. Metallibs,
AIR bodies, and source shards must never be committed.

Harvest retains non-entry AIR modules separately instead of discarding them. These modules carry
visible/intersection function implementations and other linkable helpers, along with the set of
parent metallib hashes in which each sanitized module appeared. They are not shader queue rows and
cannot be authored as standalone cases; later function-table linking resolves them as dependencies
of a stage entry across its complete parent-library provenance. Resolution succeeds only when the
symbol identifies one exact retained module across that union.

A harvest can keep only some stages: `corpus-harvest --only-stages object,mesh` keeps the modules
whose `!air.*` entry lists name one of them, and skips before any `llvm-dis` every metallib whose
function table lists no function of those stages. A full harvest may instead go to an archive, a
separate corpus root outside the checkout that is never synced, indexed, or committed; then
`corpus-harvest --from-archive ARCHIVE` merges only the selected rows from the archive's shards
into the corpus, so a later selection needs no second walk of the machine. The selection is
`--only-stages LIST`, `--rows FILE` (the source rows whose `air_sha256` is the first field of a
line of FILE), or both, and each selected row brings the archive's library modules of its parent
libraries, the dependencies linking may resolve it against. A `--rows` file naming a row the
archive does not hold (or one the stage filter beside it refuses) fails before anything is merged.

## Cases

One AIR can have several named cases. A case explicitly records buffers, threadgroup memory,
acceleration structures, top-level/array/argument-buffer textures, samplers, color render targets,
an optional depth/stencil attachment, vertex
inputs, kernel stage inputs, function constants, dispatch or draw parameters, selected output,
comparison, and execution-safety contract. Missing bytes, sizes, formats, dimensions, constants, or
output choices are errors. AIR the checker does not classify as mechanically loop-free (a CFG or
call-graph cycle among reachable functions, an unresolved call, or bitcode the typed decoder
refuses) uses `execution_safety: authored_bounded`;
its rationale must identify the authored input or function-constant value that makes every
reachable loop finite. `loop_free` is verified, never trusted (see `docs/VALIDATION.md`).

Fragment cases declare every reflected color output in `render_targets`, every reflected depth or
stencil aspect in `depth_stencil`, and provide a `draw`.
Both executors derive a matching fullscreen vertex interface from AIR metadata, preserve the
authored initial attachment bytes with load semantics, execute the draw, and compare the selected
color, depth, or stencil region. The AIR depth qualifier (`any`, `less`, or `greater`) determines
the pipeline depth comparison on both APIs. This is the same observation/evidence path used by
kernel cases.

Custom `air.imageblock_master` fragments additionally declare `fragment_imageblock` planes keyed
by exact AIR user semantic and product-supported member format; a fragment-imageblock output
selects that semantic and a 2D region.

Vertex cases declare `vertex_inputs`, exactly one render target at index zero, and a
`vertex_observation` selecting position or a reflected user-varying location. The generated
fragment observer makes that vertex result an exact render-target observation on both APIs.

An instance `acceleration_structures` entry represents a literal instance acceleration structure. Its
`child_references` length is the instance count. The Metal oracle builds that many identity
instances of a canonical triangle and binds the resulting opaque object. Candidate runners bind
the translator's documented shadow ABI at the same Metal buffer index: little-endian `u32`
instance count, reserved zero `u32`, then the listed little-endian `u64` host pointer payloads.
Those pointer payloads are deliberately explicit; raw child-pointer identities are not portable
Metal-versus-Vulkan comparison values.

A primitive entry sets `kind: "primitive"` and supplies `primitive_triangles_b64` as tightly packed
little-endian `float3` vertices (36 bytes per triangle). The Metal oracle builds the BLAS from those
vertices; the Vulkan candidate uses the identical bytes in the primitive geometry shadow.

Direct Metal visible-function references are authored under `visible_function_references` as the
logical metadata function name plus the SHA-256 of its retained AIR module. The module may belong
to a separately harvested library because Metal's linked-functions contract is explicitly
cross-library; both the Metal oracle and Vulkan candidate consume that same exact module. Visible
and intersection function tables explicitly list their Metal buffer binding and populated slots.
A visible slot names an exact function plus the SHA-256 of a retained non-entry AIR module from the
same parent metallib. An intersection slot is tagged either `linked` with that same exact
dependency contract, or `opaque_triangle` with the sorted Metal intersection-signature flags.
Opaque triangle is a populated native sentinel, not a null slot and not a synthetic AIR function.
Case checking resolves linked dependencies once. Metal binds the corresponding native table entry;
the Vulkan candidate statically replaces visible calls and fully opaque-triangle ray queries only
when their authored signatures match. Unresolved modules, cross-metallib functions, missing
symbols, unauthored slots, and signature mismatches are errors or honest translation fallback.
Tables carried inside an `air.indirect_buffer` use
`argument_buffer_intersection_function_tables`; their identity is the owning buffer binding plus
field byte offset. Reflection supplies the owner parameter, field ordinal, and Metal argument index,
so Metal encoding and Vulkan static specialization consume the same AIR-derived coordinate.

`case_id` is SHA-256 over a canonical semantic JSON encoding. It includes everything capable of
changing execution or comparison and excludes the name, rationale, author, public source, JSON key
order, shard, and storage details. `input_sha256` independently binds observations to the exact
literal inputs.

A case whose AIR is a public fixture names it: `public_source` is the fixture's repository-relative
path (`crates/metal2vulkan-validation/fixtures/public/...`). `corpus-case-check --install` sets it from the AIR, and
installing an identity its slot already holds rewrites only such fields and keeps its observations.
The checker refuses a case whose named fixture no longer hashes to its `air_sha256`, so an edited
fixture is caught by its path even where the old AIR is in no shard.

Within an AIR and an entry, `name` is a unique conceptual slot: a case's slot is its module, its
entry, and its name, so a same-name case over another entry of a module is another
slot. Replacing a slot removes the old case identity and only its Metal and Vulkan
observations through a recoverable aligned-shard transaction. Explicit deletion uses
`corpus-case-check --delete-air HASH --delete-name NAME [--delete-entry ENTRY]` (the entry is
needed only when the module's cases use the name over several entries) and performs the same
targeted cascade without installing a replacement.

## Observations

A Metal slot is `(case_id, environment_id)`. A candidate slot is
`(case_id, backend, environment_id)`. Requalification replaces the exact slot; duplicate active
slots are rejected.

Metal evidence carries case, AIR, input, output, environment, and oracle-ABI hashes. Candidate
evidence additionally carries the exact Metal output hash, SPIR-V hash, backend environment, and
executor ABI. Reuse requires every dependency to match.

Metal qualification runs a checked literal case three times. All selected outputs must be
identical, and the result must differ from the exact initial bytes across the entire selected
region. A returned poison region is a qualification failure, not a golden. Candidates require an
exact usable Metal observation and compare only against that observation's selected bytes.

## Public fixtures

`crates/metal2vulkan-validation/fixtures/public/` holds owned synthetic fixtures, not corpus data. Its top-level
`.ll` files (except `<name>.air.ll`, the typed-pointer Metal half of a fixture) are indexed as
public source rows. The product's fixture inventory
(`crates/metal2vulkan/tests/public_fixture_expectations_engine.tsv`) compiles every `.ll` file here and in
`engine/`.
A module's `air_sha256` is the SHA-256 of the whole `.ll` text, comments included, so an edited
fixture is a new AIR: its cases are reinstalled under the new hash and their observations
re-recorded. A module that declares one entry point is one row with that identity;
a row of a module that declares several is keyed by the module and its entry (below). Every
committed `.ll` here and below (again except `<name>.air.ll`) must assemble with `llvm-as` of the
LLVM that `engine/TOOLCHAIN` pins, the oldest the gates use
(`crates/metal2vulkan-validation/tests/fixtures_assemble_with_the_pinned_llvm.rs`).

`crates/metal2vulkan-validation/fixtures/public/engine/` holds owned MSL for the re-architecture. Its top-level
`.ll` files (again except `<name>.air.ll`) are indexed as public source rows too, labelled
`public/engine/<name>.ll`, so they can carry authored cases; the Metal oracle loads such a
row's `.air.ll` half. Neither the index nor the inventory descends further
(`engine/companions/` holds no rows). A
public row is one entry point of one module: a module whose `!air.kernel`,
`!air.vertex`, and `!air.fragment` lists declare several entries is a row per entry, each with the
whole module's text and `air_sha256` and keyed by `entry_row_key` (the SHA-256 of the sorted JSON
`{"air_sha256", "entry"}`), and each row's facts are read from its entry's projection of the
module. `corpus-index` and the authoring-capability census print `public-sources rows=<n>
modules=<m>`. A case over such a module names the module's `air_sha256` and its entry, and is
indexed under that entry's row.
`scripts/regen-air-fixtures/` compiles each `.metal` with `xcrun metal` and writes an
`.ll` review companion (`llvm-dis`), an `.air.ll` typed-pointer text that `xcrun metal-as`
reassembles into the same module (checked on every run), and `TOOLCHAIN`, which records the
compiler and SDK versions and the pinned LLVM (21, the oldest the gates use) whose `llvm-dis`
writes each `.ll` and whose `llvm-as` must accept it. A run writes the compiled `.air` only where
the directory already holds one: a handful of fixtures commit theirs for tests to read as bitcode,
and every other fixture's stays in the run's scratch directory. Tests that need typed-pointer
bitcode assemble it from the committed `.air.ll` with `xcrun metal-as` on
macOS and skip with a visible message elsewhere; the opaque-pointer path is covered by
`llvm-as` of the committed `.ll`.

Committed device profiles for per-profile validation live in `crates/metal2vulkan-validation/profiles/`, generated
by `scripts/regen-device-profiles/` with pinned provenance.

## Safety and privacy

The checker accepts acyclic AIR as `loop_free` and cyclic AIR only as `authored_bounded` with an
explicit finite-bound rationale. The latter is an auditable semantic assertion about the literal
inputs, not a claim that a CPU timeout can cancel already committed GPU work.

Committed cases and observations may contain authored input/output bytes but no AIR bodies or
Apple-owned binaries. Git history is the audit trail for replaced cases; active shards retain no
superseded case identities.
