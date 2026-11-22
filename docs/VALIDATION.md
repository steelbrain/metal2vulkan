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
cargo nextest run -p metal2vulkan -p metal2vulkan-air -p metal2vulkan-runtime -p metal2vulkan-devtools -p metal2vulkan-harness --all-features
cargo nextest run -p metal2vulkan-validation --all-features
cargo nextest run -p metal2vulkan-conformance --all-features
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
| `--cfg m2v_cpu_icd` | Vulkan execution tests | a CPU ICD only (lavapipe) | Linux CI, with `LP_NUM_THREADS=1` |
| `--cfg m2v_gpu` | Vulkan and Metal execution tests | a real GPU | explicitly, on a host with a GPU |
| `--cfg m2v_gpu --cfg m2v_gpu_hazards` | the GPU hazards too | a real GPU | only to test that feature, by name |

`m2v_cpu_icd` and `m2v_gpu` are mutually exclusive; `m2v_gpu_hazards` adds to `m2v_gpu` and means
nothing alone. Lint the gated code without running it, in its own target directory:

```sh
CARGO_TARGET_DIR=target/gpucfg RUSTFLAGS='-Dwarnings --cfg m2v_gpu' \
  cargo clippy -p metal2vulkan-validation --all-targets --all-features -- -D warnings
CARGO_TARGET_DIR=target/gpuhazards RUSTFLAGS='-Dwarnings --cfg m2v_gpu --cfg m2v_gpu_hazards' \
  cargo clippy -p metal2vulkan-validation --all-targets --all-features -- -D warnings
```

A site that stops announcing, or any other failing assertion, still compiles: the lint sees none of
it, and only a run does.

### The validation layer under `metal2vulkan-runtime`'s device tests

Every instance `metal2vulkan-runtime`'s device tests open runs under `VK_LAYER_KHRONOS_validation`, at
one version on every device (1.4.363.0), through one test helper (`src/tests/validation.rs`). The
product holds none of it: the helper enables the layer for its own test process with the loader's
and the layer's own variables (`VK_INSTANCE_LAYERS`, and `VK_KHRONOS_VALIDATION_REPORT_FLAGS`
`error,warn`, `VK_KHRONOS_VALIDATION_DEBUG_ACTION` `VK_DBG_LAYER_ACTION_LOG_MSG` and
`VK_KHRONOS_VALIDATION_LOG_FILENAME`, one log file per process). Every device of the cfg's kind
an instance lists must name an active validation tool (`vkGetPhysicalDeviceToolProperties`), or
the test fails: the loader ignores a layer it cannot find, so a run without the layer fails rather
than passes. Once the instance is gone, the helper reads the log: any error or warning fails the
test, each line printed with the test's name. One test makes a deliberately invalid call and
checks that its log holds exactly that message. A test process opens one instance at a time, and
the variables are set before it starts a thread: run the tests one at a time per process (nextest
does; libtest's runner with `--test-threads=1`).

The run points the loader at the layer's directory, never at a system copy:

| OS | Variables |
|---|---|
| Linux | `VK_LAYER_PATH=DIR/linux` |
| Windows | `VK_LAYER_PATH` = `DIR\windows` as a Windows path, and `DIR/windows` first on `PATH` (the layer's SPIRV-Tools and C++ runtime DLLs) |
| macOS | `VK_LAYER_PATH=DIR/macos DYLD_LIBRARY_PATH=DIR/macos` (the layer's SPIRV-Tools) |

`scripts/validation-layer make DIR --linux-from HOST --macos-from PREFIX` makes `DIR`, a git
repository a job can carry: Linux's layer copied from a host whose distro package is that version
and unaltered (`pacman -Qkk`), Windows's from MSYS2's clang64 packages, each checked against the
SHA-256 the package database lists, and macOS's from a Homebrew install at that version. Each
directory's `VERSION` says where it came from.

**A GPU hazard** is a device test that, on purpose, faults a real GPU (a GPU write into read-only
host pages), holds one for long (a command buffer past about a second, or a dispatch that spins on
a word the GPU may never see the host raise, which only its cap ends), or allocates toward the
device's limits (multi-GiB buffers, heaps or no-copy ranges, `max_buffer_length`). On Apple's GPUs
the recovery from a fault fails other processes' command buffers in flight (`.config/nextest.toml`
has the measurement). So on a GPU a hazard compiles only under `m2v_gpu_hazards`, which no sweep,
lane, stress loop or gate may pass: `--run-ignored all` and test filters cannot reach it. Under
`m2v_cpu_icd` it compiles and runs as any device test, since a CPU ICD cannot take a GPU down. Its
doc comment says which hazard it is. Run one only to test that feature, by name, alone:

```sh
CARGO_TARGET_DIR=target/gpuhazards RUSTFLAGS='--cfg m2v_gpu --cfg m2v_gpu_hazards' \
  cargo nextest run -p metal2vulkan-validation --all-features -E 'test(=<the test>)'
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

### The whole device lane in one command

The device lane is what `--cfg m2v_gpu` adds: the delta between the workspace's test list with the
cfg and without it. `scripts/device-lane/device-lane` derives that delta from two
`cargo nextest list` runs, shards it by area, runs the shards one after another, and holds the run
to the lane. Its builds set `RUSTFLAGS` to `--cfg m2v_gpu` alone, so the lane holds no GPU hazard.

```sh
scripts/device-lane/device-lane --profile debug                      # the whole lane
scripts/device-lane/device-lane --profile release --list-only        # derive and prove, no device
scripts/device-lane/device-lane --profile release \
  --filter 'test(/^replay::rt_replay::/)'                            # a slice of it
```

- **`--profile debug|release` is required**, and the report records it. Debug keeps the product's
  `debug_assert`s, release is what a timing-sensitive test needs, and neither profile's result
  stands for the other.
- `--list-only` opens no device. `--filter EXPR` intersects a nextest filterset with the lane, and
  the report then says the coverage is of that slice alone, never of the lane. An expression that
  leaves the whole lane selected narrowed nothing, and stops the run.
- Each shard is compiled by its own `cargo nextest run --no-run` pass first, so its run has nothing
  to compile and its time is the tests' alone; a sibling cargo on the same target directory can
  invalidate a build between two shards. The lane assumes the GPU is its own for the run: on a
  shared host, run it under the host's exclusive GPU hold.
- The cfg and plain legs keep separate target directories (`--cfg-target`, `--plain-target`),
  since `RUSTFLAGS` is part of a build's fingerprint. `--help` lists every option.

The shard set is derived on every run and never stored: a stored set grows no term for a test added
later, and would skip it in silence. A shard names its tests one by one, because a module holds
GPU-free tests beside device ones and a module prefix would reach past the lane.

The report (`$OUT/report.txt`, beside each shard's log, the two listings and `shards.tsv`) says:

| Line | What it means |
|---|---|
| the census | how many tests each build lists, the lane between them, how many of the lane are `#[ignore]`d, and the lane per test binary |
| `PROOF HOLDS` | every selected test is in exactly one shard, no shard reaches a test outside the selection, and the cfg build lists every test the plain build does. `PROOF FAILED` names what is adrift and stops the run before a device opens |
| a shard line | its exit status, the tests nextest ran, the tests it had to run, the seconds its build took outside the wrap, and the seconds the wrapped half took |
| the two cost lines | what the wrapped halves took in all and over how many turns, which is their turn in the host's queue and their run together and no finer, since the wrap does the waiting; then nextest's own figure for the tests, and what building cost outside the wrap |
| `COVERAGE HOLDS` | every shard ran exactly the runnable tests it holds, by nextest's own count and not the script's. nextest's `skipped` figure says nothing here: it counts every test of the shard's binary the filterset left out as well as the `#[ignore]`d ones |
| `VERDICT` | the line to read: every shard passed and the run covered the selection, or which of the two did not hold |

The lane's `#[ignore]`d tests are no part of a run of it: the cost and timing measurements and
the child halves of the tests that fork. The lane
lists them and runs none, so what each shard had to run is its tests but those; one of them runs
when a person asks for it by name.

The run exits 0 only when the proof holds, the coverage adds up and every shard passed. It is no
part of a gate: it opens a device, and takes the host's GPU a shard at a time for the length of the
lane. The lint above is what a pre-landing check can see of the cfg; an assertion needs this run.

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

## The matrix

`scripts/matrix/matrix run SHA` checks one commit on every lab GPU, on lavapipe and on the Windows
host at once, through the lab's potluck cluster, and attaches the results to the commit as a git
note under `refs/notes/matrix`: one JSON document per commit, kept in the local repository and
never pushed. `scripts/matrix/matrix show SHA` prints a note's verdicts, one line per leg.
CONTRIBUTING.md says how to run it.

A leg is one GPU, named by its PCI `vendor:device`, or `lavapipe`, or `lint`. Each GPU runs on one node, and
the nodes run at once; a node with several leg GPUs runs their device tests in turn in one job that
holds them, then times each GPU in a job of its own.

| Leg | Runs | Built |
|---|---|---|
| Each GPU of a Linux node | `metal2vulkan-runtime`'s tests and the replay player's stream device tests (`metal2vulkan-validation`'s `replay::rt_replay`) under `--cfg m2v_gpu`, under the validation layer, the GPU chosen by `VK_DRIVER_FILES` and, on Mesa, `MESA_VK_DEVICE_SELECT='vendor:device!'`; then, in a job that holds the node alone (`--exclusive`) and times this one GPU, the harness's timed workloads (`crates/metal2vulkan-harness`) on the runtime (`harness --runtime new`), the leg's suite `harness` | Once per commit on a CPU node: the tests as a nextest archive, the harness as a release build; both carried to the GPU nodes, which only run them. The archive holds its build's paths, so every node lays the tree out at the same per-commit path. Its binaries run only where the builder's do: a node of another CPU builds its own, in a job that holds no GPU |
| The Windows host's GPU | The same tests (under the validation layer) and harness, built on the host under MSYS2 CLANG64 (the pinned nightly's `x86_64-pc-windows-gnullvm` toolchain) | On the host, in a job that holds no GPU |
| `lavapipe` | The gate's suites and the device tests (`metal2vulkan-runtime`'s under the validation layer) in one build under `--cfg m2v_cpu_icd` (`cargo nextest run --workspace`, then the doctests), `LP_NUM_THREADS=1` | In place, on the builder |
| `lint` | The gate's `cargo fmt --all --check`, clippy with `-D warnings` plain, under `--cfg m2v_gpu` and under `--cfg m2v_cpu_icd`, and Rustdoc with `-D warnings`, each its own suite | In place, on the builder, beside its other jobs |

A leg passes when every suite passed and no rule below fails. Its line names the device and
driver as `vulkaninfo` reported them, the node, the tests passed of those run, the records, the
base it was compared with and the job.

Every job that runs `metal2vulkan-runtime`'s device tests unpacks the validation layer's carry (the
config's `validation_layer`, made by `scripts/validation-layer`) in its workspace and points the
loader at its OS's directory; the leg's note gives the layer's version (`layer`). A job without
the layer says so there, which is a problem of the leg, and points the loader at no layer, so its
device tests fail. The harness's timing jobs run without the layer.

Where a job names a Vulkan driver's library in `M2V_DIRECT_DRIVER` (on macOS, Mesa's Mac driver,
`libvulkan_kosmickrisp.dylib`), `metal2vulkan-runtime`'s device tests load it themselves and give it
to the loader alone through `Instance::with_drivers`, as a consumer that ships its driver does;
the loader's `VK_DRIVER_FILES` and its own search then go unused, and each test's output says
`direct driver PATH`. Elsewhere the job's `VK_DRIVER_FILES` chooses the driver.

### Records and rules

A test records a value by printing one line on its standard output; the matrix collects the lines
of passing and failing tests alike:

```text
M2V-MATRIX <kind> <test>/<name> <value>
```

The value is an unsigned integer (`time`: microseconds, decimals allowed), and a key appears once
per leg. The base is the nearest
first-parent ancestor whose note holds the same leg. Times, and the harness's own records (keys
starting `harness-`), are compared only when the base's `crates/metal2vulkan-harness` tree is this
commit's: the harness's warm-up, windows and workloads are part of the measurement, so a commit that
changes the harness sets the base for the next.

| Kind | What it counts | Fails when |
|---|---|---|
| `metric` | A deterministic count per frame or per command buffer: submissions, passes, barriers, bytes copied, pipelines created; and the replay player's totals of a play (`replay-rt/STREAM/RANGE/NAME`, [REPLAY-STREAM.md](REPLAY-STREAM.md) section 10), each one two plays of one stream at one commit on one card gave alike, printed only by a play that finished its range and skipped nothing | It is larger than the base's; a replay total, when it differs from the base's either way, or when the base has it and the leg does not (`EQUAL_METRICS`) |
| `instructions` | User-space CPU instructions per draw or per dispatch, and per harness run, the p50 over the row's timed runs: the whole process's (`harness-<runtime>/<row>/per_run`), and where the runtime names its threads, theirs with the harness's thread (`.../own_per_run`) and the rest, a driver's own threads (`.../others_per_run`) | It is more than 1% over the base's; the harness's counts are judged by the instructions gate instead (below) |
| `gauge` | Live objects of one resource kind, the key ending in the point it was read at: `@100` and `@200` (loop iterations), `@drop` (after the device dropped) | `@drop` is not 0, or `@200` differs from `@100`; for the memory pool's `blocks`, which frees allocations an earlier workload left idle once the most bytes in use over its window falls and the device is quiet for 20 ms, only `@200` over `@100` |
| `time` | A duration in microseconds: what a program waits on in a harness workload's run, timed on the host: for the frame rows (`frame`, `frame_5cb`), their throughput, two runs in flight, from one run's last completed handler to the next's, and their latency from idle as rows of their own (`<row>_from_idle`), one run in flight, from its first commit to its last completed handler; for another workload that times its command buffers, its last command buffer's commit-to-complete time (the others' lateness is a column of the harness's table and no record), else the host wall time a run adds (a run's GPU time is a record of its own, `harness-gpu-<runtime>/<row>`, parked here) | Only as a cross-GPU outlier: its ratio to the base's is more than 1.25 times the median of that ratio over at least 3 GPUs. A leg with such an outlier of a harness row's time has its harness timed once more after every leg is in, and an outlier fails the leg only where both passes agree |
| `size` | The size a harness row whose size follows the device ran at (`long_4cb`) | Never: it says what work the row's times measured |
| `locks` | The lock budget's locked instructions per million added commands (`lock_budget/draw`, `lock_budget/bound_draw`, `lock_budget/dispatch`; rt's `tests::locks`), the whole process's in user space, on a node whose matrix setup names its CPU's event in `M2V_LOCK_EVENT` (`cpu_core/r21d0`, `r1e25`) | The test fails past 0.25 per added command, on every node alike; on a node whose setup names the event, a GPU leg missing any of the three fails; elsewhere each budget test passes counting nothing and prints `M2V-MATRIX uncounted lock_budget/<key>`, which the leg's line shows as `not counted` |

Every cache has a test that fills it past its capacity and records its gauge.

Two kinds of harness record are parked (`PARKED_RECORDS` in `scripts/matrix/results.py`): the
warm first-use records and the rows' GPU times, both below. A parked gauge's change between `@100`
and `@200`, or a parked time's cross-GPU outlier, is named on the leg's line and fails no leg; a
parked gauge's `@drop` still fails.

The harness's first-use records (`harness-new/<set>/first_use_*`: how many of the runtime's
first pipeline uses waited, and for how long) depend on the driver's shader cache, which outlives a
run. The harness times the runtime's rows in one process, `harness --runtime new`, with a fresh
empty cache directory (`MESA_SHADER_CACHE_DIR` for Mesa, `__GL_SHADER_DISK_CACHE_PATH` for NVIDIA;
on the Windows host, `LOCALAPPDATA` and `USERPROFILE` point into it, which the Intel driver's cache
follows), and a fresh empty store for the runtime's own pipelines (the harness's `--store`), so its
first uses and its `create_state` row are cold. A short second run on the same directory and store
follows (`--runs 5 --warmup 1 --no-settle`), and its first-use records are kept as
`harness-new-warm/<set>/first_use_*`, the same first uses with both warm; they are parked.

Both runs skip the creation rows (the harness's `--skip 'create_*'`): those run after them in a
process of their own, `--only 'create_*'`, on a fresh empty cache directory and store, so no row
before them warms the driver for them; that process's device records are kept as
`harness-<runtime>/creates/device/...`, apart from the other process's. Each creation row's first
sample is a fresh object's on a device that has gone that way once: the harness makes a warm-up of
the row's kind first (a buffer of the same length, kept; a compute state of another kernel, with
its dispatch; a render state of another fragment function), so what a device pays once, which a
guest that keeps its device pays once in its life, stays out of the sample. A row whose size
follows the device (`long_4cb`) is sized by the first run, and the later runs, warm and alone ones
included, are given that size (the harness's `--size`); each run records it as
`size harness-<runtime>/<row>`.

A harness row's time is judged by the `time` rule alone, as a cross-GPU outlier, and a row whose
time was recorded more than once fails, with no one time to show. The frame and burst rows
(`WALL_ROWS` in `scripts/matrix/results.py`), where the runtime counted them (Linux, a node
that grants the counter), must also pass the instructions gate, on the runtime's own
instructions per timed run: those of the threads it started, which `Gauges::threads` names, and of
the harness's thread (`instructions harness-new/<row>/own_per_run`). A driver's own threads are
left out (`.../others_per_run`): they swing by millions of instructions per run between processes
(NVIDIA's on `burst_160`, 0 to 25 million), where the runtime's own do not. The own count is at
most 1.03 times its stored count (`scripts/matrix/instructions.tsv`: one count per row where the
legs agree within 3%, else one per driver family). A stored count only ratchets down: a leg more
than 1% under it is named, and nothing raises it without a ruling; a row with none stored is named.
A run that records its whole count but not its own fails the row. Every row records its
counts; only these are gated on them.

Each row's GPU time, the p50 over its timed runs of a run's first GPU start to its last GPU end, is
recorded as `time harness-gpu-<runtime>/<row>` where the row has one; it is parked, shown and not
judged. `run` and `show` print, under each leg's line, every harness row's time, its GPU time and,
where the leg counted them, its instructions per run (with the runtime's own).

The golden hash table's translations do not move. The table compiles through
`Device::from_queries`. When its test reports a cell that changed under an unchanged
key, or one under a changed profile fingerprint, the leg fails with `GOLDEN TABLE MOVED`. The one
exception is a range where a commit since the table's last bless has a subject that starts with
`old-runtime regression fix`. New, gone and re-keyed fixture rows only need a bless.

`metal2vulkan_devtools::instructions::InstructionCounter` measures `instructions`: the calling
thread's user-space instructions from Linux's `perf_event_open`, the form an unprivileged process
may open at `perf_event_paranoid` 2. On a hybrid x86 CPU it sums the event over each core type's
unit, since the generic event counts on one of them only.
`metal2vulkan_devtools::instructions::ProcessInstructions` measures the harness's: one counter per
core type opened with `inherit` before the harness makes a device, so it counts every thread the
runtimes and their drivers start. `metal2vulkan_devtools::instructions::ThreadInstructions`
measures the runtime's own: one counter per core type on each thread it names, opened by its OS
id before a row's timed runs. Off Linux none opens, so the Windows leg records no
`instructions`; a node that refuses them records none either.

The rules and the sizing of the matrix's jobs have their own tests, which need no cluster:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/matrix
```

## The Linux gate in a container

`scripts/linux-gate` runs CI's Linux job, or any command, in a container: Ubuntu 26.04 pinned by
digest with CI's packages, plus `VK_LAYER_KHRONOS_profiles` built from a pinned
commit. It is where CPU-ICD execution runs; the container never sees a host GPU.

```sh
scripts/linux-gate --cpus N [--slot NAME] [--platform amd64|arm64]            # ci.yml's Linux job
scripts/linux-gate --cpus N [--slot NAME] [--platform amd64|arm64] \
    [--icd lavapipe] [--profile P] \
    [--private-corpus DIR] [--scratch DIR] -- <command> ...                   # any command
```

- The container gets 4 GiB of memory per CPU. Each `--slot` keeps its own target volume; remove a
  slot's volumes when you are done with it.
- The default command, `scripts/linux-gate-ci`, runs every Linux step of `ci.yml` in order: change
  the two together.
- `scripts/qualify-public-cases` (also run by CI) runs the public cases on lavapipe against their
  committed Metal rows, and fails on anything but a match.
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
| `authoring-capabilities [--reclassify-all]` | For every indexed AIR identity, whether a case can express and both executors can run its structure (`ToolingRequirement`s). Closed only when `classified == total`, `remaining == 0` and `unresolved == 0`; a warm rerun reads no source shard. A row Metal cannot build (`metal_refused`) or the runtime refuses, typed (`runtime_refused`: a requirement whose limit is `Limit::Runtime`, each held both ways by a lavapipe device test where the runtime has a call for it, mesh pipelines by the runtime README's known gap) is closed without that executor's run and counted apart; a tooling gap of its own still keeps it open |
| `translation --profile P...` | Each row translated for each committed profile, recorded as a typed outcome class under the current product fingerprint (`--current-fingerprint`, `--retry-failures`, `--hash-file PATH` select rows). It fails when a row breaks a ceiling |
| `budgets --profile P...` | The budget gates from recorded rows: no row above 20 s or 500 MiB, no stopped worker, and each row's ledger peak at least 0.7 of its allocator peak |
| `specialization --profile P...` | Rows translated under planned function-constant value sets and the pipeline inputs their AIR states, not only with none |
| `authored-compile [--bless \| --bless-additions \| --case ID]` | Every committed authored case compiled, with the requests its executor builds, as the runtime compiles them (a kernel for its dispatch's threadgroup size, a draw with no binding plan's variant), on each profile of `corpus/authored-compile.tsv`; it fails on any changed cell. The table is re-blessed on `development` |
| `bitcode-reader` | The bitcode reader's decode compared with `llvm-dis`, and the intrinsic census |
| focused audits (`rasterization-rate-maps`, `sample-rate`, `ulong-textures`, `fragment-varyings`, `linked-functions`, `tile-functions`, `mesh-draws`) | One structural family, selected by current facts, translated in batches |

The translation, specialization and authored-compile audits, and `corpus-pipeline`, compile for a
committed profile as the runtime's device describes it (`engine::runtime_description`): read at
the runtime's Vulkan version and restricted to what the runtime enables of it
(`engine::runtime_enables`, of `metal2vulkan_runtime::floor::enable_set`). Every Vulkan context
the executor opens enables the same set. It is the runtime's tables applied to what the profile
states, which differs from the runtime's own answer for a device in four places:

- The runtime enables the border swizzle's two features together or not at all; a profile that
  states `borderColorSwizzle` without `borderColorSwizzleFromImage` (the NVIDIA ones) keeps
  `borderColorSwizzle` here, where the runtime enables neither.
- The runtime enables the attachment feedback loop only where the device's descriptor limits hold
  the attachment reads; this keeps it wherever the profile states it.
- A profile below the runtime's floor (`moltenvk-m3`, which states `nullDescriptor` and `logicOp`
  false) is compiled for the set's intersection with what it states, not for the runtime's answer,
  which is to refuse the device.
- A context at API 1.2 (every context without a profile restriction) reaches the features the
  runtime's version has as core through their extensions, and has none that only that version
  brings without an extension.

```sh
cargo run -p metal2vulkan-validation --release --bin corpus-triage -- \
  --audit authoring-capabilities --limit 100000
lockf /tmp/m2v-index.lock cargo run --release -p metal2vulkan-validation --bin corpus-triage -- \
  --audit translation --profile runtime-floor --profile moltenvk-m3 --current-fingerprint --limit 500 --summary-only
cargo run --release -p metal2vulkan-validation --bin corpus-triage -- \
  --audit budgets --profile runtime-floor --profile moltenvk-m3 --summary-only
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
class) of every public fixture entry point on every committed profile. A behavior-preserving
change keeps every cell; a change that moves one bumps `SEMANTICS_VERSION` and re-blesses
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
```

- **Metal qualification** (`corpus-metal`) runs the case three times through Metal and records its
  observation for the environment. `--prepare-only` does the CPU side and stops before a device
  opens, to screen cases.
- **The Vulkan candidate** (`corpus-vulkan`, on Linux) translates the
  case through the public API in a killable compile worker, executes exactly what the binder plans,
  and compares the observations with Metal's. Every context runs under the validation layer and
  enables what the runtime enables of the device: no buffer robustness, and the image robustness
  the device states.
- **Through the runtime.** On Linux, without a profile restriction, every kernel case the engine
  maps (buffers, device buffer arrays included, bound textures and samplers, argument buffers of
  buffers, textures and samplers, threadgroup memory, function constants and linked functions,
  dispatched by threads and observing a buffer, a texture, an argument-buffer member or nothing)
  runs through `metal2vulkan-runtime`'s public API instead, as a consumer runs it. Each texture is
  made as the Metal oracle makes it, Shared and written with `replace_region`, at an index the run
  grants. An argument buffer holds each member's `gpuAddress` or `gpuResourceID` at its field, as
  a program writes them, and each member buffer and texture is used for reading and writing
  (`use_resource`). A 32-bit float depth texture, which the runtime takes no upload of, is filled
  by a draw of the owned `depth_fill` pair, one pass per level and slice, as a guest fills one; a
  case that gives a depth image a depth outside `[0, 1]`, which a draw clamps, carries
  `depth_outside_unit_range`, the runtime's limit on every device, and runs on Metal alone. An
  out-of-bounds case's facts are read before the state is made, from each variant the translator
  lowers of the kernel on the runtime device's own translator (`Device::translator`, under
  `LoweringInputs::kernel_variant`); variants that contain a target differently refuse the case,
  `not_supported{uncontained:variant ...}` naming the one that differs. The facts are the identity
  domains' (`DevicePolicy`): a buffer reached through a pointer loaded from an argument buffer is
  the address the pointer holds, which nothing checks (the runtime's README states it), so such a
  case is refused as `not_supported{uncontained:unchecked loaded_pointer}`, and an argument
  buffer's texture array is the descriptor arrays its resource ids' masked indices read
  (`resource-id:shader`), a read past it a texel of a bound texture or zero. Each buffer target,
  and the buffer a texture-buffer target is made over, sits between guards of its declared reach
  in a Shared block of its own, read back after the dispatch. A multisampled texture, which neither Metal nor the runtime uploads, is Private and
  filled by a draw per slice of the owned `sample_fill` pair, its fragment run once per sample
  writing that sample's texel from a buffer, a color one through an unsigned-integer view of its
  texel size so its bits cross unconverted, a depth one under an always-passing depth test that
  writes. A cube array is made as any cube, of its cubes' faces. A bound sparse
  texture carries `sparse_texture`, which the runtime does not make; a bound texture buffer runs
  on both executors (the runtime makes it over its buffer's bytes); a kernel's function table carries `kernel_function_table`, its imageblock (a
  tile function's included) `kernel_imageblock`, its acceleration structure or
  intersection-function table `kernel_acceleration_structure`, its stage inputs or stage-in region
  `kernel_stage_input`, and a rasterization-rate map, a render pass's or one whose data a buffer
  holds, `rasterization_rate_map`: each the runtime's limit, run on Metal alone. The runtime's
  indirect command buffer holds kernel-encoded indexed draws that inherit the pipeline state and
  bind their own buffers; one of compute commands carries `indirect_compute_commands`, one that
  allows other draws `indirect_draw_types`, one whose commands set their own pipeline state or
  inherit the encoder's buffers `indirect_command_inheritance`, a command the host encodes
  `host_encoded_indirect_commands`, and a pipeline state an argument buffer holds, which the
  runtime gives no `gpuResourceID`, `argument_buffer_pipeline_state`: each the runtime's limit,
  run on Metal alone. A command buffer of the runtime's form carries `indirect_indexed_draws`, a
  gap of this executor, which maps no indirect command buffer, so the checker takes no case of
  it. A pixel format the runtime makes no image of on the device, and a sample count it makes no
  texture of (`Limit::TextureSamples`), records `not_enabled`, as the executor's own context
  records a format or a sample count its query lacks. A vertex or fragment case of any
  sample count draws through the runtime the same way (`engine::runs_render_through_runtime`): its own
  stage's buffers, bound textures and samplers, argument buffers of buffers, textures and samplers
  (made, written and used as a kernel's), function constants and linked functions, its
  generated companion, its vertex buffers, its color targets (Shared 2D
  textures, loaded and stored by the pass), a depth/stencil attachment of one aspect as the oracle
  makes it (a Shared `Depth32Float` filled by a draw of its depths, as a 32-bit float depth texture
  is, or a Shared `Stencil8` written from its bytes, loaded and stored by the pass; in a pass of
  more samples each attachment Private and multisampled, filled as a multisampled texture is, a
  stencil aspect of one value in every sample cleared to it by the pass, and every sample read
  back after the draw by a dispatch of the pair's kernels, unresolved, in the layout
  `executor_contract::multisample` states), the oracle's
  depth-stencil state, cull mode and winding, and one direct or indexed draw of primitives or one
  patch draw of a post-tessellation vertex function with no patch index buffer (its factors a
  Shared buffer bound as the tessellation factor buffer at the oracle's offset and instance
  stride), observing a buffer, a texture, an argument-buffer member, a color target, the
  attachment's aspect or nothing, its out-of-bounds targets, if any, attribute buffers. Their facts are read
  before the state is made, from the translator's compile of the state's descriptor on the
  runtime device's own translator: the runtime's floor enables no buffer robustness, so a
  fixed-function vertex fetch past its buffer is checked nowhere and the case is refused as
  `not_supported{uncontained:unchecked vertex_input}`. A rasterization-disabled vertex case renders
  into the oracle's 1x1 `R8Unorm` attachment, and a fragment case of no attachment into the
  oracle's pass of none: `renderTargetWidth` and `renderTargetHeight` 1, `defaultRasterSampleCount`
  the case's samples. Layered rendering, a patch index buffer or an out-of-bounds target other
  than an attribute buffer keeps the case on the executor's own context. A vertex buffer's dynamic stride carries `dynamic_vertex_stride` and a fragment
  function's imageblock `fragment_imageblock`, each the runtime's limit. An indexed
  draw past its index buffer, which the runtime refuses as Metal's validation does
  (`Refusal::IndexBufferOffset`), carries `index_buffer_past_end`, and a depth/stencil attachment
  of both aspects, the oracle's two textures, which the runtime renders into in no one pass yet
  (`Gap::RenderPass` of `DepthStencilViews`), carries `separate_depth_stencil`: each the runtime's
  limit, run on Metal alone. Its device worker enables the
  validation layer through the loader's and the layer's variables, and any message the layer logs
  fails the row. The pipeline state's make is held to 500 MiB of peak live allocations and 20 s.
  Its row records the state's precision notes, the runtime's fingerprint and an out-of-bounds
  case's facts, and no SPIR-V, status, subgroup shape, module containment or artifact. The comparison bounds `ReductionOrder` by the
  simdgroup float sums and products the case's AIR calls, the operations the translator's census
  names for that note.
- **The verdict.** A row whose bytes equal Metal's, or differ within its comparison class's
  bound, is a `Match`. A difference is never a match: under a class unbounded by precision notes
  it is a `NotedDifference` (the notes can explain a difference; whether they explain this one is
  not checked), counted in its own `noted_difference` column, and otherwise a `Mismatch`. A
  difference of a case the C3 reviewed list holds under an answer gap its structure meets stays a
  `Mismatch`, its row and batch line naming the gap (`held_open_by`): the user's decision of
  2026-10-06 keeps answer gaps open until an exact emulation exists.
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
one driver, with its environment ID, committed profile and devices. Today: lavapipe on arm64 and
x86 (the public cases), and the AMD Radeon 780M under RADV, Intel's Raptor Lake Xe graphics under
ANV, and the NVIDIA GeForce RTX 5080 Laptop GPU (every Metal-qualified case).
`cutover/c3-references.json` names the Metal environment that is the reference (the Apple M3 Max),
with its declared identity; each case is judged against its row.

`corpus-status --c3` judges every venue's slot of every case: `match`, `reviewed`,
`device_gap`, `out_of_scope_ray_tracing`, `reference_gap`, `mismatch`, `refused`, `stale` or
`missing`; a noted difference reads `mismatch` there, since C3 asks for a match. A row recorded
on a device or driver its venue does not allow is rejected. A change that moves a venue's profile
fingerprint without changing any artifact is handled by rekeying its records
(`corpus-status --venue-keys`, then `--rekey-venues`); a change that moves artifacts needs a
re-record on that venue.

### Command replay

- `command-replay play` plays a replay stream ([REPLAY-STREAM.md](REPLAY-STREAM.md)) on a Vulkan
  device through the runtime and writes its report: the runtime's counters for each group of
  command buffers it waited for, each frame and the whole play, the census of written words that
  look like addresses (they play as recorded, but those in a buffer's range the stream noted
  escaped, which move with it) and the address arguments it moved,
  and a SHA-256 of each checked buffer or texture against the recorded one. `command-replay dump`
  prints a stream's calls.

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
