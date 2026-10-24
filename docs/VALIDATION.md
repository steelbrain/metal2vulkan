# Validation

How to check that the translator and the runtime give Metal's answers. Two contracts are kept
apart:

- **Emitter stability**: the SPIR-V the translator emits for owned fixtures does not change
  unless the change means it to (the golden output hashes).
- **Semantic correctness**: an authored case gives the same observations on Vulkan as on Metal.

A valid SPIR-V module proves neither. Choose the smallest route that proves the claim:

| Claim | Starting point |
|---|---|
| An ordinary code change | The gate below |
| A behavior-preserving translator change | The gate, and unchanged golden output hashes |
| An intentional change of output | A synthetic regression test, the reviewed golden drift (and a `SEMANTICS_VERSION` bump), and the affected authored cases rerun on Metal and Vulkan |
| A corpus-wide support or performance claim | A resumable indexed `corpus-triage` audit, with its completion, read and time summaries |

Validation lives in the unpublished crate `crates/metal2vulkan-validation`. Paths below are
relative to it unless they start with `crates/`, `docs/` or `scripts/`. Each tool's `--help` lists
every option.

## The gate

```sh
cargo fmt --all
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo nextest run -p metal2vulkan -p metal2vulkan-air -p metal2vulkan-runtime -p metal2vulkan-devtools --all-features
cargo nextest run -p metal2vulkan-validation --all-features
cargo test --workspace --all-features --doc
```

None of these opens a device. They use the repository's own synthetic fixtures; private AIR and
GPU runs are optional and explicit.

## Device tests and GPU safety

A shader that never terminates keeps a GPU busy after its process is killed, and a wedged GPU can
hang the host. So everything that opens a device is explicit, bounded and refusable.

**Device tests compile only under a cfg** (not a feature, since the gate builds with
`--all-features`):

| cfg | Adds | Device | Run |
|---|---|---|---|
| none | GPU-free tests only | none | the gate |
| `--cfg m2v_cpu_icd` | Vulkan execution tests | a CPU ICD only (lavapipe, SwiftShader) | Linux CI, with `LP_NUM_THREADS=1` |
| `--cfg m2v_gpu` | Vulkan and Metal execution tests | a real GPU | explicitly, on a host with a GPU |

The two are mutually exclusive. Lint the gated code without running it, in its own target
directory:

```sh
CARGO_TARGET_DIR=target/gpucfg RUSTFLAGS='-Dwarnings --cfg m2v_gpu' \
  cargo clippy -p metal2vulkan-validation --all-targets --all-features -- -D warnings
RUSTFLAGS='-Dwarnings --cfg m2v_gpu' cargo nextest run -p metal2vulkan-validation --release
```

- **Device selection is an argument.** Every device-opening binary takes `--device-type cpu|gpu`,
  with no default. `cpu` matches only a CPU device and never falls back to a GPU; `gpu` takes the
  loader's first GPU. `corpus-metal` takes only `gpu`.
- **Waits are bounded.** Every host wait on device work goes through `src/gpu/wait.rs` (10 seconds
  by default; `--gpu-wait-timeout SECONDS` raises it for one command). A test fails if an
  unbounded wait appears elsewhere.
- **The hang tripwire.** When a wait times out, or a submission ends with a device fault, the
  process writes `~/.cache/m2v-gpu/hang` (JSON: what ran, on what, when), frees nothing and exits
  with status 86. While the file exists every GPU open is refused; a CPU device never checks it.
- **If a hang is suspected** (the tripwire, a device-lost, a display blink, an unresponsive UI):
  stop launching GPU work; read the tripwire; check for other GPU users on the host
  (`pgrep -fl '[c]argo'`), since another workload may fault the GPU on purpose; check the system
  log for GPU restarts; reproduce the suspect module on a CPU ICD only, where a stuck shader is an
  ordinary killable process; add a deterministic regression; then a person removes the tripwire.

**The runtime lane** (`runtime_lane::` tests) runs the runtime on a device under
`VK_LAYER_KHRONOS_validation`. `METAL2VULKAN_LANE_VALIDATION=sync` (or `load-store`) adds the
layer's synchronization checks to every lane device and prints each failing message as a
`LANE-VALIDATION <test>: ...` line:

```sh
METAL2VULKAN_LANE_VALIDATION=sync RUSTFLAGS='--cfg m2v_cpu_icd' cargo nextest run \
  -p metal2vulkan-validation --all-features --no-fail-fast -E 'test(/^runtime_lane::/)'
```

A lane device opens whether or not the loader can provide the layer it asks for: the runtime drops
a layer the loader answers `VK_ERROR_LAYER_NOT_PRESENT` for and says so as the device is made. A
claim of what the layer reported therefore states what it holds, against the layers the instance
enabled and nothing else (`runtime_lane::layer_reported_nothing`): where the layer was dropped the
claim is not exercised, and the test prints `LANE-LAYER-ABSENT <test>: ...` and a count per message
list instead of passing on an empty list. A host whose layer manifest names its library by a bare
filename (which the loader looks for on the system library search path) provides no layer at all,
and a lane run there is evidence for everything but the layer's checks. The executor's own Vulkan
context refuses to open without the layer instead (`candidate::platform::device`).

**Seeded differential sequences** (`parity::sequences`): 64 seeded, race-free Metal programs of
render passes, blits, dispatches, indirect commands, fences and events over buffers, heaps and
textures, run on the runtime and compared with committed Metal results
(`corpus/observations/parity/sequences.jsonl`) and with a CPU model. A failing seed is shrunk and
printed. Metal's results are recorded on a Mac by the ignored test
`parity::sequences::lane::metal_recording::record_metal`, each seed in a child process of its
own. A seed whose child Metal's driver crashes is recorded as a Metal-side gap, with the crash's
signature (the call, the faulting frame and the command); the lane then has nothing of Metal's to
compare that seed with, runs it on the runtime all the same (it must complete with no refusal and
no validation message), and names it in its output. Under `METAL2VULKAN_LANE_VALIDATION=sync` the
lane reads each hazard the layer reports against the sequence: the layer takes a shader's buffer
as accessed over its whole descriptor (the bytes the shader reaches widened to 16-byte granules,
or from the bound offset to the end where it indexes the buffer), so a hazard between two
commands that share no byte of its memory changes no answer, and prints as a `SEQUENCE-SYNC` line
naming what Metal orders between them (nothing, for untracked memory between encoders or a
concurrent encoder's dispatches). The lane matches the commands by the Vulkan command the layer
names and by the command buffers it may have checked since the lane last took its messages
(after each commit). A hazard between commands that share a byte (the generator orders those, so
the runtime must), on a buffer no command of the sequence bound, or of another shape fails the
seed. So does one the layer's words fit both to such a pair and to pairs whose bytes are apart;
the minimized sequence then tells them apart: when it still needs a command of an apart pair,
the pair sharing bytes alone raised no hazard. Two seeds fail so today (7 and 50, on lavapipe,
the RTX 3090 and RADV), each settled that way: the hazard is between untracked commands in two
encoders, which Metal leaves unordered and whose bytes are apart, and the layer's ranges meet
because a texel kernel's or a draw's buffer binds from its offset to the buffer's end, as Metal's
does.

**Generated-program fuzzers** (`metal2vulkan_devtools::fuzz`: structured, irreducible, inline,
access, workgroup, pointer select) run device-free in the translator's tests
(`crates/metal2vulkan/src/tests/fuzz_interp.rs`) on its SPIR-V interpreter, at every subgroup
width, against each generator's reference model. A longer soak:

```sh
METAL2VULKAN_TEST_FUZZ_SOAK='--seeds N [--first S] [--profile P] [--threads 32|64] [--jobs J]' \
  cargo nextest run --release -p metal2vulkan --all-features --run-ignored only fuzz_soak
```

Their device variants run under `--cfg m2v_gpu`:

```sh
RUSTFLAGS='-Dwarnings --cfg m2v_gpu' cargo nextest run --release -p metal2vulkan-validation \
  --all-features -j 1 -E 'test(/_fuzz::.*_on_the_device/)'
```

## The Linux gate in a container

`scripts/linux-gate` runs CI's Linux job, or any command, in a container: Ubuntu 26.04 pinned by
digest with CI's packages, plus SwiftShader and `VK_LAYER_KHRONOS_profiles` built from pinned
commits. It is where CPU-ICD execution runs; the container never sees a host GPU.

```sh
scripts/linux-gate --cpus N [--slot NAME] [--platform amd64|arm64]            # ci.yml's Linux job
scripts/linux-gate --cpus N [--slot NAME] [--platform amd64|arm64] \
    [--icd lavapipe|swiftshader] [--profile P] \
    [--private-corpus DIR] [--scratch DIR] -- <command> ...                   # any command
```

- The container gets 4 GiB of memory per CPU. Each `--slot` keeps its own target volume; remove a
  slot's volumes when you are done with it.
- The default command, `scripts/linux-gate-ci`, runs every Linux step of `ci.yml` in order: change
  the two together.
- `scripts/qualify-public-cases` (also run by CI) runs the public cases on lavapipe in both
  robustness legs against their committed Metal rows, and fails on anything but a match.
- lavapipe's vector width follows the host (128 bits on arm64, 256 on x86 with AVX2);
  `LP_NATIVE_VECTOR_WIDTH=256` forces the wider one.

## The private corpus

Optional, local and never committed: AIR harvested from metallibs on a Mac, kept as JSONL shards
under `corpus/local/`, with a disposable SQLite index (`corpus/.index.sqlite`) of identities, byte
locations and cached facts. [corpus/README.md](../crates/metal2vulkan-validation/corpus/README.md)
describes the formats.

```sh
cargo run -p metal2vulkan-validation --release --bin corpus-harvest   # macOS
cargo run -p metal2vulkan-validation --bin corpus-index
cargo run -p metal2vulkan-validation --bin corpus-status
```

- **Harvest** merges new rows only into the shards they belong to, and creates no obligations.
  A row is one entry point of one module; library modules that only supply linked functions are
  kept apart (`corpus/local/library-modules`).
- **Indexing is incremental**: an unchanged shard is never opened, and a warm index reads no
  source bytes. `--rebuild` is for recovery only; `--check` verifies the index against the shards.
- Sweeps that write into the index run under `lockf /tmp/m2v-index.lock <command>`.
- Tools built in a worktree default their corpus root to that worktree; pass `--corpus DIR` to use
  another checkout's.

### Audits

Every audit is a `corpus-triage --audit ...` run over indexed rows, each attempt in a killable
worker under the 20-second and 500-MiB ceilings, resumable in bounded batches (`--limit N`,
`--after HASH`), with `--jobs` defaulting to the host's logical CPU count.

| Audit | What it answers |
|---|---|
| `authoring-capabilities [--reclassify-all]` | For every indexed AIR identity, whether a case can express and both executors can run its structure (`ToolingRequirement`s). Closed only when `classified == total`, `remaining == 0` and `unresolved == 0`; a warm rerun reads no source shard |
| `translation --profile P...` | Each row translated for each committed profile, recorded as a typed outcome class under the current product fingerprint (`--current-fingerprint`, `--retry-failures`, `--hash-file PATH` select rows). A fragment function of more colors than the profile draws at once is translated as its render pipeline's color groups, one module a group, each checked by L3. It fails when a row breaks a ceiling |
| `budgets --profile P...` | The budget gates from recorded rows: no row above 20 s or 500 MiB, no stopped worker, and each row's ledger peak at least 0.7 of its allocator peak |
| `specialization --profile P...` | Rows translated under planned function-constant value sets and the pipeline inputs their AIR states, not only with none |
| `authored-compile [--bless \| --bless-additions \| --case ID]` | Every committed authored case compiled, with the requests its executor builds, on each profile and leg of `corpus/authored-compile.tsv`; it fails on any changed cell. The table is re-blessed on `development` |
| `bitcode-reader` | The bitcode reader's decode compared with `llvm-dis`, and the intrinsic census |
| focused audits (`rasterization-rate-maps`, `sample-rate`, `ulong-textures`, `fragment-varyings`, `linked-functions`, `tile-functions`, `mesh-draws`) | One structural family, selected by current facts, translated in batches |

```sh
cargo run -p metal2vulkan-validation --release --bin corpus-triage -- \
  --audit authoring-capabilities --limit 100000
lockf /tmp/m2v-index.lock cargo run --release -p metal2vulkan-validation --bin corpus-triage -- \
  --audit translation --profile vk12-floor --profile moltenvk-m3 --current-fingerprint --limit 500 --summary-only
cargo run --release -p metal2vulkan-validation --bin corpus-triage -- \
  --audit budgets --profile vk12-floor --profile moltenvk-m3 --summary-only
```

Other corpus tools:

- `corpus-pipeline --device-type cpu|gpu --profile P` asks a Vulkan driver to create each
  translated pipeline, without dispatching anything (driver acceptance).
- `--profiles-layer P` (on `corpus-vulkan` and `corpus-pipeline`) runs every context through
  `VK_LAYER_KHRONOS_profiles` restricted to a committed profile, to test a profile on a device that
  exceeds it.

## Authored cases

A case is a JSON manifest that names one AIR (by hash) and every resource, initial byte, function
constant, dispatch or draw, output region and comparison rule. Runners do not infer, seed, resize,
repair or reinterpret anything. Cases live in `corpus/cases/`, Metal and Vulkan observations in
`corpus/observations/`, keyed by the case's semantic ID. The schema and its resource kinds are in
[the crate's README](../crates/metal2vulkan-validation/README.md); they include out-of-bounds
targets for containment checks (`out_of_bounds`), sparse textures (`sparse`), fragment, vertex,
tessellation and tile cases, object and mesh cases (`mesh_draw`),
argument buffers and indirect command buffers.

```sh
cargo run -p metal2vulkan-validation --bin corpus-index
# Read an indexed AIR function and write one manifest for it.
cargo run -p metal2vulkan-validation --bin corpus-case-check -- --manifest /path/to/case.json --install
cargo run -p metal2vulkan-validation --bin corpus-index
cargo run -p metal2vulkan-validation --bin corpus-status
```

- Editing a manifest replaces its `(air_sha256, name)` slot; observations of the old identity are
  removed.
- After a change to the AIR layer, reflection, the descriptor ABI or the case schema, re-check every
  installed case: `corpus-case-check --recheck-all` (needs the private sources). CI re-checks the
  cases over public fixtures.

## Contract 1: emitter stability

`crates/metal2vulkan/src/tests/golden_hashes.rs` pins a hash of the final SPIR-V (or the refusal
class) of every public fixture entry point on every committed profile, and, in a second table, of
every fixture that samples under every exact form (Metal's sampler emulated and
`DevicePolicy::exact_answers`). A behavior-preserving change keeps every cell; a change that moves
one bumps `SEMANTICS_VERSION` and re-blesses
([CONTRIBUTING.md](../CONTRIBUTING.md#golden-output-hashes-and-semantics_version)). The fixture
inventory (`crates/metal2vulkan/tests/public_fixture_expectations_engine.tsv`) pins which
fixtures compile or refuse on which profile. Over the private corpus, the translation audit records
outcomes, not bytes.

## Contract 2: semantic correctness

Each case runs first on Metal, then on Vulkan, and the observations are compared.

```sh
cargo run -p metal2vulkan-validation --release --bin corpus-metal -- \
  --case-id CASE --environment-id METAL_ENV --device-type gpu
cargo run -p metal2vulkan-validation --release --bin corpus-vulkan -- \
  --case-id CASE --environment-id VENUE --device-type cpu|gpu
cargo run -p metal2vulkan-validation --release --bin corpus-moltenvk -- \
  --case-id CASE --environment-id VENUE --device-type gpu     # macOS
```

- **Metal qualification** (`corpus-metal`) runs the case three times through Metal and records its
  observation for the environment. `--prepare-only` does the CPU side and stops before a device
  opens, to screen cases.
- **The Vulkan candidate** (`corpus-vulkan` on Linux, `corpus-moltenvk` on macOS) translates the
  case through the public API in a killable compile worker, executes exactly what the binder plans,
  and compares the observations with Metal's. Every context runs under the validation layer, in a
  robustness leg: the stated leg enables every robustness feature the device states, and the
  `robustness-off` leg none, so both containment encodings run.
- **Batches.** `--batch --out OUT` runs every selected case, one process per case, each killed
  whole after `--run-timeout`, resuming from `OUT/done.tsv`; `--batch-report` and
  `--batch-install STORE` read and install the results. A batch starts no GPU run while the
  tripwire exists.
- **What counts.** An observation is current only for the product, executor and runtime versions
  it was made with: a row records the product fingerprint, the executor ABI and `RUNTIME_VERSION`,
  and a change of any makes it stale until it is rerun. A stored observation, a valid module or a
  manifest alone is not evidence.
- **No-effect declarations and reference gaps.** A case may declare that an output equals its
  authored bytes, and a Metal pipeline creation that fails naming one feature is recorded as a
  reference gap rather than a pass.
- **What it cost.** Every row records what its case cost in time: how many executions the figure
  covers, what they took in all, and the least one of them took, in microseconds. A Metal row's
  figure is its three executions, a candidate row's its one run on the device, and a refusal's the
  attempt it refused on, each read by the clock the pipeline-creation ceiling is checked against.
  The store refuses a row or a refusal that records none, so the figure is there for every case
  either side runs, whether it produced output or a typed refusal. It is not semantics: no verdict,
  digest, fingerprint or census key reads it, two rows that differ only in their figures are one
  row, and a row recorded before the figure existed carries none.

### Venues

`cutover/c3-venues.json` lists the venues the cases must pass on: each is one translation key under
one driver, with its environment ID, committed profile, devices and the legs it records. Today:
MoltenVK on Apple's M3 Max, lavapipe on arm64 and x86, the AMD Radeon 780M under RADV, Intel's
Raptor Lake Xe graphics under ANV, and the NVIDIA GeForce RTX 5080 Laptop GPU.
`cutover/c3-references.json` names the Metal environment that is the reference (the Apple M3 Max),
with its declared identity; each case is judged against its row.

`corpus-status --c3` judges every venue's slot of every case, in both legs: `match`, `reviewed`,
`device_gap`, `out_of_scope_ray_tracing`, `reference_gap`, `mismatch`, `refused`, `stale` or
`missing`. A row recorded on a device or driver its venue does not allow is rejected. A change that
moves a venue's profile fingerprint without changing any artifact is handled by rekeying its
records (`corpus-status --venue-keys`, then `--rekey-venues`); a change that moves artifacts needs a
re-record on that venue.

### Draw A/B and command replay

- `corpus-draw-ab` draws a logged render pipeline's fragment function through the runtime on seeded
  inputs and records what it writes and how long it takes; `corpus-draw-ab compare --base DIR
  --change DIR --draws FILE --runs N` compares two builds. A draw skipped in both legs is not
  compared, and fails the comparison unless named in `--expected-skips`.
- `command-replay` runs one recorded stretch of Metal commands through the runtime from a manifest
  that names every object and byte.
- `format-scout` drives every Metal pixel format through the runtime's public API, one use at a
  time, and records what each use did.

## The product bug loop

1. Say whether the SPIR-V should change, and name the structural feature.
2. Reproduce with one AIR or an owned synthetic fixture.
3. Add a synthetic regression test where practical.
4. Fix the product, without names or environment gates.
5. Run the gate's format, clippy and the narrow tests.
6. Check the golden drift: only the intended cells move.
7. Where cases cover the change, rerun them on Metal (only if the Metal side changed) and on the
   affected venues.

## What stays private

The published crates exclude validation, docs, scripts and every capture. Committed validation
data is owned synthetic fixtures, authored manifests and hash-identified observations. Never commit
`corpus/local/`, the index, metallibs, AIR, SPIR-V dumps, third-party shader bodies, or anything
derived from a consumer's guest shaders.
