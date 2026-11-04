# AGENTS.md

Rules for any AI agent working in this repository. Follow them unless the user overrides one for
the current task.

## The project

metal2vulkan runs Metal shaders and Metal programs on Vulkan. It is a Cargo workspace (a virtual
manifest at the root) of six crates under `crates/`:

| Crate | Published | What it is |
|---|---|---|
| `crates/metal2vulkan` | yes | The translator: Metal AIR to SPIR-V through a Metal-shaped API (device, library, pipelines, binding plans), and the `metal2vulkan` command (feature `cli`) |
| `crates/metal2vulkan-runtime` | yes | The Metal API on Vulkan, built only on the translator's public API |
| `crates/metal2vulkan-air` | yes | The AIR front end: container, bitstream, bitcode, typed IR, metadata ABI, intrinsics, canonicalization, `graph`, `budget` |
| `crates/metal2vulkan-validation` | no | Authored semantic cases, Metal and Vulkan execution, harvest, the corpus index and audits |
| `crates/metal2vulkan-conformance` | no | The Metal conformance suite: one case run on Metal and on the runtime, recorded and compared field by field. Independent of the harvested corpus |
| `crates/metal2vulkan-devtools` | no | The external-tool runner, fixture assemblers and fixture paths the tests share |

The three published crates share one version; `metal2vulkan` pins `metal2vulkan-air` exactly
(`=0.x.y`). License: dual, `GPL-2.0-or-later OR LGPL-3.0-or-later`, at the recipient's choice. The
root `LICENSE` is the notice; `LICENSE-GPL-2.0`, `LICENSE-LGPL-3.0` and `LICENSE-GPL-3.0` (which the
LGPL v3 incorporates) are the texts. Each published crate carries copies of all four, held equal to
the root's by a test (`published_crates`).

The translator's core lives in `crates/metal2vulkan-air/src/` and, in
`crates/metal2vulkan/src/`, `memory/`, `lower/`, `structurize/`, `spirv/`, `pipeline/` and
`bind/`. Prefer structural fixes there over one-off workarounds.

| Doc | Role |
|---|---|
| `README.md` | What it is, install, command line, quick library use |
| `docs/API.md` | The translator's API guide; the degrade-notice catalogue; accepted differences |
| `crates/metal2vulkan-runtime/README.md` | The runtime's guide |
| `docs/HOWTO.md` | Integration: tools, caching, failures, isolation |
| `docs/ARCHITECTURE.md` | The translator's stages and module map |
| `docs/FAST-PATHS.md` | The ledger of speed-for-accuracy defaults and their exact switches |
| `docs/VALIDATION.md` | The validation ladder |
| `docs/CONFORMANCE-COVERAGE.md` | Generated, along two axes: which Metal items the conformance suite compares against Metal, and which of each Metal enum's and option set's values it compared |
| `CONTRIBUTING.md` | Build, test, golden tables |
| `crates/metal2vulkan/src/env_vars.rs` | The `METAL2VULKAN_*` variables |

## The owner's two rules

These override anything below.

1. **The GPU's native arithmetic; fast by default, exact on request for the rest.** Arithmetic
   and value answers (`float` and `half` maths, NaN bits, minimum and maximum, rounding, texel and
   colour conversions, sampling and filtering, resolves) are the device's own: where the device
   has a native form, it is the only form, recorded as a `PrecisionNote` where it can differ from
   Metal's. Never add a path or a switch that emulates Metal's bits where the device has a native
   form. Capability stays: what the device lacks is implemented or emulated (Translation honesty,
   below). Every other choice that trades accuracy for speed (pipeline-creation time or GPU time)
   defaults to the fast form. Its accurate form stays behind an opt-in `DevicePolicy` switch
   (`DevicePolicy::exact_answers()` selects them all), with tests of both settings. Never delete
   either form of such a choice, and never make its accurate form the default: that is the
   owner's call alone. `docs/FAST-PATHS.md` is the ledger, one row per choice, edited in place (a
   ledger, not a journal). A commit that adds, removes or changes such a choice, its bound or its
   switch updates its row in the same commit; `tests::fast_paths` checks the ledger against the
   code.
2. **An accuracy note is logged, never fatal.** A precision note or degradation is recorded and
   announced, and the work runs, on every device. Only what Metal itself refuses, or an operation
   that cannot run at all, may fail or end anything.

## Translation honesty

- **Structure over names.** Decide lowering from IR structure (types, address spaces, access
  chains, AIR's metadata ABI), never from a function, type, variable or shader name. Dispatching on
  the stable `air.*` and `llvm.*` intrinsic families is allowed; prefer a structural test when one
  exists. A name-keyed branch that passes one shader while failing identically shaped others is a
  defect. No behavior keyed to an app, a shader hash or a corpus census.
- **Honest FALLBACK.** Unsupported input returns a typed `Err` (the command line prints
  `FALLBACK`). Never emit wrong-but-valid SPIR-V to silence a gate.
- **Scope is the device's.** Refuse only what the device truly lacks; implement or emulate the
  rest (an arithmetic or value answer is the device's native one: rule 1). A near-equivalent, the
  device's native answer included, is allowed only when the artifact records it (a
  `PrecisionNote`) and the runtime announces it.
- **Unknown stays unknown.** Do not invent semantics for unsupported opcodes, types or layouts.
  Fail visibly and leave a test or a doc note.
- **No env-gated product paths.** Translation never branches on environment variables. The
  `METAL2VULKAN_*` variables are for tool paths, operational settings and default-off
  diagnostics only.
- **Fix causes, not observations.** A probe may explain a failure; committed behavior needs a
  structural or ABI reason.

## Claims and evidence

- A claim is only as broad as its evidence. One verified path does not license "all", "never" or
  "zero". Scope replies, commits and docs to what you ran.
- Say **verified** only for commands you ran and results you saw; say **expected** or
  **unaudited** for the rest.
- Prefer the smallest testable change. When you fix a bug class, add a test or a measurable check
  that fails without the fix.
- Evidence for a gated or device-specific path must show the path ran (a counter, a log line, an
  assertion), not only that a run passed.

## Translation ceilings

- **20 seconds per translation attempt, end to end**: handing the AIR to an isolated worker,
  translation, SPIR-V validation and reporting. Success or an honest FALLBACK must arrive within
  it. A timeout is a failure to diagnose and fix, never a way to classify supported input.
- **500 MiB per translation attempt**, enforced at the worker boundary (peak live allocations and
  resident growth). On a breach, kill the worker's whole process group and remove its scratch
  files. Do not raise the ceiling without a new explicit instruction from the user.
- Keep total use bounded: parallelize only independent rows, cap workers and queues, account for
  `workers × 500 MiB`. No unbounded threads, processes, queues, caches or retained IR.
- Corpus tools default `--jobs` to the host's logical CPU count; pass `--jobs N` only when asked.
- Remove work rather than hide it: incremental, shard-local indexing; parse only what is selected;
  cache immutable analysis; never reuse analysis across a semantics-changing rewrite.
- Changes to translation, planner, indexing or validation hot paths need release-mode evidence:
  wall time and peak memory for the largest relevant row and a bounded representative batch.
  Report the slowest row and the scope. Debug timings prove nothing.
- An over-budget translation is an active bug, never accepted debt. Add a deterministic check for
  the redundant work behind it.

## Validation coverage

Validation lives in `crates/metal2vulkan-validation`; [docs/VALIDATION.md](docs/VALIDATION.md)
has the commands.

- **Close the census.** The authoring-capability census
  (`corpus-triage --audit authoring-capabilities`) covers every indexed AIR identity. A fresh
  census is closed only when `classified == total`, `remaining == 0` and `unresolved == 0`.
  Reviews, allowlists, older observations or a smaller selection do not resolve a gap.
- **One vocabulary.** `ToolingRequirement` (`src/requirement.rs`) and the structural checks in
  `src/executor_contract/` are the contract between classification, the case checker, the Metal
  and Vulkan executors, stored facts and reports. Do not grow parallel string lists. Supporting a
  structure means updating the schema, validation, both executors, classification and tests
  together; what one side cannot execute exactly stays an unsupported requirement.
- **Describe families apart from gaps.** A focused `AuditTarget` selects current structural facts,
  not rows that carry a requirement, so fixing a capability does not empty its regression audit.
- **The index is an acceleration structure.** Refresh only new or changed shards; a warm,
  unchanged audit reads no source shard; an exact-row lookup reads only its own shard.
- **Cache validity follows semantics.** Fingerprints cover everything that can change
  classification, translation or expected output. Bump the matching version when behavior changes
  (`SEMANTICS_VERSION`, `ARTIFACT_FORMAT_VERSION`, `RUNTIME_VERSION`, the executor and census
  versions) and recompute through the indexed path.
- **Tests own their inputs.** Index, cache and audit tests build their own synthetic shards; they
  never depend on a private corpus, an existing index, filesystem order or cache warmth.
- **The right proof for the claim.** After classifier changes, a full
  `--reclassify-all` census, then a warm pass with zero source reads. After a feature change, its
  focused audit over the whole selected family. After translation changes, a fresh translation
  census where the corpus is available. Corpus-wide claims need corpus-wide results.
- **Authored evidence is execution.** Qualify a case on Metal, run the Vulkan candidate and compare
  the observations. A stored observation, a valid module or a manifest alone proves nothing.

## Tools and scratch files

- The libraries run no external tool. The command line runs `llvm-as` (for `.ll` input) and
  `spirv-val`. Tests and validation also use `llvm-dis`, `spirv-as` and `xcrun metal-as`. Each is
  found on `PATH` or through `METAL2VULKAN_<TOOL>`.
- Remove scratch files under the OS temp dir as soon as the tool is done with them; no long-lived
  dumps under fixed `/tmp` paths. FALLBACK repro bundles under `$TMPDIR/metal2vulkan-repros` (or
  `METAL2VULKAN_REPRO_DIR`) may stay.
- Keep disk writes down (CONTRIBUTING.md, "Many trees on one machine"):
  - make worktrees with `scripts/lean-tree worktree` (the observations are clones) and source
    snapshots with `scripts/lean-tree snapshot` (no observations unless the tree runs validation);
  - keep each worktree's targets inside it (`target/`, and `target/gpucfg` for `--cfg m2v_gpu`):
    kache shares incremental state only between in-tree targets; start a new one from a seed with
    `scripts/seed-target`;
  - builds are incremental, and compile through `scripts/rustc-wrapper` (kache where it is
    installed: a local cache that also carries incremental state between worktrees);
  - record Instruments traces through `scripts/xctrace-run`, and delete the `.trace` once
    exported.
- `pgrep -f` and `ps | grep` match their own command line. Always bracket the first character:
  `pgrep -f '[c]orpus-refresh'`.

## What not to commit

- Apple-owned binaries, disk images, mined metallib or AIR corpora, private golden sets.
- Anything derived from a consumer's guest shaders: shader caches, metallibs, translated SPIR-V or
  MSL, logs with shader text, census results over them.
- `crates/metal2vulkan-validation/corpus/local/`, and raw `*.metallib`, `*.air`, `*.spv` or
  `*.spvasm` dumps (gitignored).
- AIR or SPIR-V of anyone else's shader. The exceptions are our own: `.air` that
  `scripts/regen-air-fixtures` compiles from MSL we wrote, and synthetic `.spvasm` we wrote,
  committed under `crates/metal2vulkan-validation/fixtures/public/`.
- `target/`, `Cargo.lock` (ignored: this tree is library-first), `.cache/`, secrets, large binaries,
  name-keyed special cases.

Committed and allowed: synthetic fixtures under `crates/metal2vulkan-validation/fixtures/public/`,
authored cases under `corpus/cases/` and observations under `corpus/observations/` (no AIR bodies
in either).

## Layout and ownership

- Product logic stays in `crates/metal2vulkan`. The runtime reaches it only through its public
  API, which the compiler enforces. Validation and devtools use public APIs too; a test that needs
  an internal item lives in the crate's own `src/tests/`.
- Validation may depend on MoltenVK or a Vulkan ICD; none of that enters a published crate's
  dependencies.
- `crates/metal2vulkan/src/spv.rs` (the SPIR-V grammar items the translator names) and
  `crates/metal2vulkan/src/spirv_binary/table.rs` (the test parser's table, test-only) come from
  `scripts/regen-spirv-grammar/`, from Khronos's grammar JSON alone; never hand-edit them. A
  newly named opcode or enumerant goes in that script's `items.txt`, then regenerate.
- The published crates (`metal2vulkan`, `metal2vulkan-air`, `metal2vulkan-runtime`) take only
  dependencies that offer a GPL-2.0-compatible license (MIT, BSD, ISC, Unlicense or a similar
  option); never one that is Apache-2.0-only, GPL-3.0-only or LGPL-3.0-only. The dual license's
  GPL-2.0 option must hold for consumers that link the crates into GPL-2.0 programs. Code in
  their sources is ours or comes from such a source with its notice kept.
- The pipeline interface: link numbers set 0 densely in `ResourceSlot` order; set 1 holds the heap
  arrays (docs/API.md section 5).
- A new or changed refusal, `RuntimeGap` or public type regenerates `docs/REFUSALS.md`
  (`python3 scripts/refusal-surface/refusal_surface.py`) and updates the `api_surface` table.
  Public docs never link to private items.
- Every public runtime item has a doc comment that names its Metal counterpart and what differs
  or is refused.

## Build, test, CI

Tests run under [cargo-nextest](https://nexte.st) at its default parallelism. nextest skips
doctests, so they have their own line. The gate:

```sh
cargo fmt --all
cargo clippy --workspace --all-targets --all-features -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --all-features --no-deps
cargo nextest run -p metal2vulkan -p metal2vulkan-air -p metal2vulkan-runtime -p metal2vulkan-devtools --all-features
cargo nextest run -p metal2vulkan-validation --all-features
cargo nextest run -p metal2vulkan-conformance --all-features
cargo test --workspace --all-features --doc
```

- MSRV: `rust-version` in the crates' manifests (1.98). The tree builds on the dated nightly
  `rust-toolchain.toml` pins, with rustc's parallel front end on every core (`-Zthreads=0`,
  `.cargo/config.toml`).
- Warnings are errors everywhere (CI sets `RUSTFLAGS=-Dwarnings`). Never hide one behind
  `#[allow(...)]`; fix it.
- Features: `serde`, `ash` and `cli` (which enables `serde`). Every gate passes `--all-features`;
  CI also builds `metal2vulkan` with no feature and with each alone.
- Device tests compile only under `--cfg m2v_gpu` or `--cfg m2v_cpu_icd`
  ([docs/VALIDATION.md](docs/VALIDATION.md)).
- CI (`.github/workflows/ci.yml`) runs on `ubuntu-26.04` and `macos-26` (never `*-latest`): format,
  clippy (all features, each feature alone, `--cfg m2v_gpu`), Rustdoc, the API reference check
  (`scripts/api-docs`), packaging, the suites, doctests, and on Linux the device tests and public
  case qualification on lavapipe.

When chasing a hang, read the whole nextest output; do not pipe it through `head` or `grep`.

## Code style

- **No placeholder ownership tricks.** Never move IR, modules, functions, blocks, instructions or
  analysis out of a borrowed owner with `mem::take`, `mem::replace` with an empty value, `swap`
  with an empty value or a helper that does the same. Consume and return the owned value, borrow
  it, or restructure around an explicit result. Remove an existing occurrence when you touch it.
- Run `cargo fmt --all` before every commit; keep clippy clean.
- Match the surrounding style; prefer small files and structural fixes over special-case tables.
- No dead code: delete what nothing uses.

## Docs and comments

- **No planning documents and no design citations.** Code, comments and docs cite no design-record
  entries (`D<n>`), work-package codes or TODO section numbers, and the tree keeps no plan, status
  or journal. The why of a change goes in its commit message; a durable contract goes in the doc
  that owns it.
- **The honest trade-offs stay current.** `docs/FAST-PATHS.md` (speed-for-accuracy choices, device
  gaps and known gaps), `docs/REFUSALS.md` (generated), `docs/METAL-COVERAGE.md` (generated),
  `docs/CONFORMANCE-COVERAGE.md` (generated) and
  `docs/API.md`'s degrade-notice catalogue and accepted differences change in the same commit as
  the behavior they describe. An open gap is recorded where a user reads it: a typed refusal, a
  degrade notice or precision note, or `docs/FAST-PATHS.md`'s known gaps.
- Consumer-visible API or command-line changes update `README.md` and `docs/`.
- Docs say what is true now, plainly; no history.

## Git workflow

- Commit at every coherent step: a focused subject, and a body saying what, why and how it was
  verified.
- Before each commit run `cargo fmt --all`, and nothing more.
- **Never work on or advance `master` unless the user explicitly asks for it.** Work happens on
  `development` (create it from `master` if it does not exist) or on your own branches off it.
  Only the user's explicit request moves `master`: no commit, merge, fast-forward, reset or push.
- **Merge into `development` at once**, with no pre-merge test, queue or gate. On a conflict, merge
  `development` into your branch, resolve it there, commit, and merge again. `development` may be
  broken for a while; a claim about a change still waits for its evidence.
- **Checks run on `development` after the merge**: fmt, clippy in every configuration, Rustdoc,
  the suites, the public-surface table, the authored compile census and the golden tables. The
  golden tables and the compile census are re-blessed on `development`, not in the change.
- **Fix forward.** Every failure the checks report is a work item for the change's owner, fixed by
  a new commit. Never amend, rebase published history, squash, force-push or revert by rewriting
  history. Never use `git stash`.

## The loop

1. **Scope** the smallest testable increment and name how you will verify it.
2. **Read** the code and docs it touches; assume nothing.
3. **Change** it, keeping product paths free of env gates and name keys.
4. **Test** it with the test or measurable check it needs.
5. **Document** it where a durable contract changed (and in `docs/FAST-PATHS.md` for any
   speed-for-accuracy choice).
6. **Commit and merge** into `development`, then fix forward what `development`'s checks report.

**Done** means merged into `development`, `development`'s checks green for it, claims matching
evidence, translations in the touched scope within 20 seconds and 500 MiB, and capability audits
still closed.
