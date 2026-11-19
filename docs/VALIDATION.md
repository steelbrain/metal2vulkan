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
cargo nextest run -p metal2vulkan -p metal2vulkan-air -p metal2vulkan-rt -p metal2vulkan-runtime -p metal2vulkan-devtools -p metal2vulkan-harness --all-features
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

### The validation layer under `metal2vulkan-rt`'s device tests

Every instance `metal2vulkan-rt`'s device tests open runs under `VK_LAYER_KHRONOS_validation`, at
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
  cargo nextest run -p metal2vulkan-validation --all-features \
  -E 'test(=runtime_lane::host_memory::a_gpu_fault_fails_its_command_buffer_and_the_queue_runs_on)'
```

**Work past a CPU device's reach** compiles only under `m2v_gpu`: a device test whose command
buffer a CPU device cannot run within Metal's bound on one. The runtime's watchdog fails a queue's
front batch that goes 20.2 s without its timeline advancing as Metal's timed-out command buffer
(`GpuHang`, the runtime's `command_buffer/stall.rs`), on every device. The two dispatches past
2^32 threads in `runtime_lane::indirect_dispatches` are such tests: each took 24 s on lavapipe
with 16 CPU threads to itself, and 0.1 to 0.3 s on an RX 7900 XTX and a Radeon 780M.

- **Device selection is an argument.** Every device-opening binary takes `--device-type cpu|gpu`,
  with no default. `cpu` matches only a CPU device and never falls back to a GPU; `gpu` takes the
  loader's first GPU. `corpus-metal` takes only `gpu`.
- **A GPU of few channels.** NVIDIA's driver gives each `VkQueue` a channel, and a runtime device
  of the queue plan takes 18: an RTX 2080 holds 10 or 11 such devices at once across processes,
  where a 16-thread host's 16 test processes ask it for up to 288 `VkQueue`s. Past that, the driver
  refuses a device the plan's queues and the runtime makes it with its least plan's three,
  announcing `metal2vulkan-degrade-least-queue-plan`, or refuses those too
  (`Error::DeviceRefused`). A runtime lane device so made or so refused is made again, the one
  made dropped first, after a pause that doubles from 0.25 s to 2 s, for at most 120 s in all
  (`runtime_lane/device_room.rs`), each pause a `LANE-DEVICE-ROOM` line on stderr. The runtime
  crate's own device tests make theirs the same way (`src/tests/plan_room.rs`, `PLAN-DEVICE-ROOM`
  lines), and one still of its least plan past the bound fails its test; the tests of the least
  plan itself stand the driver's refusal in (`refuse_next_devices`). `cargo nextest run --profile
  few-channels` runs three device tests at a time (`.config/nextest.toml` has the measurement),
  leaving room for other programs' devices on the same GPU.
- **A busy window the host outlasted is asked again.** The `did_modify_range` oracle's steps
  (`runtime_lane/device_copies.rs`) and the host copies of new `Shared` and `Managed` textures
  (`runtime_lane/first_use.rs`) need the GPU busy past their calls, about 2 s of busy work
  against calls of 5 to 47 ms on an RX 7900 XTX and of 20 to 670 us on a Radeon 780M. A try whose busy work ended before its calls
  returned still checks its bytes, writes a `LANE-BUSY-MISS` line on stderr (each call's time and
  the busy work's GPU time), and runs again, at most 3 tries a step
  (`runtime_lane/busy_premise.rs`); a step that misses every try fails, as a call that waits for
  the GPU's earlier work would.
- **The runtime's own device tests take four `VkQueue`s, not the plan's 18.** Its test build's
  `DeviceOptions::default()` holds queue limits of a round robin of two and none for presenting
  (`TEST_QUEUE_LIMITS`, `device/create.rs`), as the runtime lane's devices hold theirs
  (`LANE_QUEUE_LIMITS`): NVIDIA's driver makes and destroys one device at a time across processes,
  each queue adding to both. A test whose work needs more (a command queue made while the test
  holds two others, a presenting, reserved or extra queue) panics where it asks for it, on every
  device, and makes its device with `queue_limits: None`.
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
  --filter 'test(/^runtime_lane::degrade_notices::/)'                # a slice of it
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

The lane's `#[ignore]`d tests are no part of a run of it: the cost and timing measurements, the
recorders (`parity::lane::metal_recording`) and the child halves of the tests that fork. The lane
lists them and runs none, so what each shard had to run is its tests but those; one of them runs
when a person asks for it by name.

The run exits 0 only when the proof holds, the coverage adds up and every shard passed. It is no
part of a gate: it opens a device, and takes the host's GPU a shard at a time for the length of the
lane. The lint above is what a pre-landing check can see of the cfg; an assertion needs this run.

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
enabled and nothing else: where the layer was dropped the claim is not exercised, and the test
prints `LANE-LAYER-ABSENT <test>: ...` and a count per message list instead of passing on an empty
list. Five forms state it, and no lane test reads a message list for emptiness any other way:

| Form | The claim |
|---|---|
| `layer_reported_nothing_at_all(&messages)` | The widest claim, and the one the lane states oftenest: whatever the layer checked of the work so far, it reported nothing of it |
| `layer_reported_nothing(&messages, claim)` | The same, of the narrower `claim` the test names, which leads the failure and the `LANE-LAYER-ABSENT` line |
| `layer_reported_nothing_but_apart_writes(&messages, claim, &writes)` (`runtime_lane::apart_writes`) | The same, but a write after write the layer reports on a buffer whose every write the test noted as it placed it (`NotedWrites`: a dispatch's, a draw's, a fill's, a bounded draw's or a patch draw's) passes in three cases. **Apart** (a `LAYER-APART-WRITES` line): two noted writes or more, each one Vulkan command's, no two sharing a byte. The layer takes a shader's storage buffer as written over its descriptor's whole range, which the runtime binds wider than the bytes the shader provably writes (from the offset alignment down, in 16-byte granules) and orders by (`sync_coverage::disjoint_words_of_one_block_are_a_hazard_to_the_layer_alone`); no order of writes that share no byte changes one. **Idle draw** (`LAYER-IDLE-DRAW`): the two `vkCmdDrawIndexedIndirect` of one draw the index bounds pass bounds, reported as recorded, one such draw noted per command buffer, on a device that announced `indirect-index-bounds-pass`: at most one of the two draws. **Idle chunk** (`LAYER-IDLE-CHUNK`): two `vkCmdDrawIndirect` chunk draws of emulated patch draws, reported as recorded, the patch draws of each command buffer noted apart, on a device that announced `emulated-tessellation-draws`: a test keeps each patch draw below `2^30` triangles and the scan writes a `vertexCount` of 0 past them, so at most one chunk of a draw draws. A read after a write, a write after a read, a copy's access, two noted writes of one byte, and either split form reported otherwise fail as above |
| `layer_reported_nothing_but_announced_stage_races(&messages, claim, &buffers)` (`runtime_lane::stage_races`) | The same, but a hazard between a later draw's vertex stage and an earlier draw's fragment stage of one pass, reported as recorded, on a buffer the test names, passes with a `LAYER-STAGE-RACE` line where the device announced the difference it is: a read after a write under `in-place-vertex-reads`, a write after a read or a write under `fragments-before-geometry`. Metal runs a pass's vertex functions before its fragments; Vulkan orders the draws only through the pass's barriers |
| `nothing_reported_without_the_layer(&messages, claim)` | A device the test made without the validation layer, to time something: the claim is the driver's own messages and the runtime's, so the layer's absence takes nothing from it. It fails where such a device did ask for the layer after all |

A host whose layer manifest names its library by a bare
filename (which the loader looks for on the system library search path) provides no layer at all,
and a lane run there is evidence for everything but the layer's checks. The executor's own Vulkan
context refuses to open without the layer instead (`candidate::platform::device`).

Whether the layer's synchronization checks see a kernel's write of a buffer it binds is asked of
the layer, never of its version: the positive control
(`runtime_lane::sync_coverage::layer_sees_kernel_writes`, one dispatch writing an untracked
buffer through the runtime's recording, then a copy of it with nothing between) runs first.
Where the control finds the layer blind, each `sync_coverage` test whose hazard needs a kernel's
write, and `parity::sequences::lane`'s hazard control, prints
`LAYER-BLIND-KERNEL-WRITES <test>: ...` naming what it did not check, and requires the layer to
have reported nothing of the rest of its run.

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

## The matrix

`scripts/matrix/matrix run SHA` checks one commit on every lab GPU, on lavapipe and on the Windows
host at once, through the lab's potluck cluster, and attaches the results to the commit as a git
note under `refs/notes/matrix`: one JSON document per commit, kept in the local repository and
never pushed. `scripts/matrix/matrix show SHA` prints a note's verdicts, one line per leg.
CONTRIBUTING.md says how to run it.

Both end with one table: every harness row over its limit on every leg (the harness budget
below), with its kind (GPU time, host time, or count for the instructions gate), its ratio and
limit, the frozen arm's verdict on it (`VOID`, `stands`, `rebased` or `no base`), and its
status, from `scripts/matrix/status.tsv` by card and row: an open record, with its owner and its
age in rounds; a known gap, with the place it was ruled; or, for a row with neither, `NEW` in the
first round it is over and `UNOWNED` after. A row of a card's own variance (`CARD_VARIANCE`)
reads as its card's known gap. The table changes no verdict.

A leg is one GPU, named by its PCI `vendor:device`, or `lavapipe`, or `lint`. Each GPU runs on one node, and
the nodes run at once; a node with several leg GPUs runs their device tests in turn in one job that
holds them, then times each GPU in a job of its own.

| Leg | Runs | Built |
|---|---|---|
| Each GPU of a Linux node | `metal2vulkan-rt`'s tests under `--cfg m2v_gpu`, under the validation layer, the GPU chosen by `VK_DRIVER_FILES` and, on Mesa, `MESA_VK_DEVICE_SELECT='vendor:device!'`; then, in a job that holds the node alone (`--exclusive`) and times this one GPU, the harness's timed workloads (`crates/metal2vulkan-harness`), both runtimes in one process (`harness --runtime new,old`), the leg's suite `harness` | Once per commit on a CPU node: the tests as a nextest archive, the harness as a release build; both carried to the GPU nodes, which only run them. The archive holds its build's paths, so every node lays the tree out at the same per-commit path. Its binaries run only where the builder's do: a node of another CPU builds its own, in a job that holds no GPU |
| The Windows host's GPU | The same tests (under the validation layer) and harness, built on the host under MSYS2 CLANG64 (the pinned nightly's `x86_64-pc-windows-gnullvm` toolchain) | On the host, in a job that holds no GPU |
| `lavapipe` | The gate's suites and the device tests (`metal2vulkan-rt`'s under the validation layer) in one build under `--cfg m2v_cpu_icd` (`cargo nextest run --workspace`, then the doctests), `LP_NUM_THREADS=1` | In place, on the builder |
| `lint` | The gate's `cargo fmt --all --check`, clippy with `-D warnings` plain, under `--cfg m2v_gpu` and under `--cfg m2v_cpu_icd`, and Rustdoc with `-D warnings`, each its own suite | In place, on the builder, beside its other jobs |

A leg passes when every suite passed and no rule below fails. Its line names the device and
driver as `vulkaninfo` reported them, the node, the tests passed of those run, the records, the
base it was compared with and the job.

Every job that runs `metal2vulkan-rt`'s device tests unpacks the validation layer's carry (the
config's `validation_layer`, made by `scripts/validation-layer`) in its workspace and points the
loader at its OS's directory; the leg's note gives the layer's version (`layer`). A job without
the layer says so there, which is a problem of the leg, and points the loader at no layer, so its
device tests fail. The harness's timing jobs run without the layer.

Where a job names a Vulkan driver's library in `M2V_DIRECT_DRIVER` (on macOS, Mesa's Mac driver,
`libvulkan_kosmickrisp.dylib`), `metal2vulkan-rt`'s device tests load it themselves and give it
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
| `metric` | A deterministic count per frame or per command buffer: submissions, passes, barriers, bytes copied, pipelines created | It is larger than the base's |
| `instructions` | User-space CPU instructions per draw or per dispatch, and per harness run, the p50 over the row's timed runs: the whole process's (`harness-<runtime>/<row>/per_run`), and where the runtime names its threads, theirs with the harness's thread (`.../own_per_run`) and the rest, a driver's own threads (`.../others_per_run`) | It is more than 1% over the base's; the harness's counts are judged by the instructions gate instead |
| `gauge` | Live objects of one resource kind, the key ending in the point it was read at: `@100` and `@200` (loop iterations), `@drop` (after the device dropped) | `@drop` is not 0, or `@200` differs from `@100`; for the memory pool's `blocks`, which frees allocations an earlier workload left idle once the most bytes in use over its window falls and the device is quiet for 20 ms, only `@200` over `@100` |
| `time` | A duration in microseconds: what a program waits on in a harness workload's run, timed on the host: for the frame rows (`frame`, `frame_5cb`), their throughput, two runs in flight, from one run's last completed handler to the next's, and their latency from idle as rows of their own (`<row>_from_idle`, held to the row's ceiling and floor), one run in flight, from its first commit to its last completed handler; for another workload that times its command buffers, its last command buffer's commit-to-complete time (the others' lateness is a column of the harness's table and no record), else the host wall time a run adds (a run's GPU time is a record of its own, `harness-gpu-<runtime>/<row>`, parked here and judged by the harness budget) | Only as a cross-GPU outlier: its ratio to the base's is more than 1.25 times the median of that ratio over at least 3 GPUs |
| `size` | The size a harness row whose size follows the device ran at (`long_4cb`) | The leg's two runtimes ran the row at different sizes: its times are then not compared (the harness budget) |
| `locks` | The lock budget's locked instructions per million added commands (`lock_budget/draw`, `lock_budget/bound_draw`, `lock_budget/dispatch`; rt's `tests::locks`), the whole process's in user space, on a node whose matrix setup names its CPU's event in `M2V_LOCK_EVENT` (`cpu_core/r21d0`, `r1e25`) | The test fails past 0.25 per added command, on every node alike; on a node whose setup names the event, a GPU leg missing any of the three fails; elsewhere the budget prints `LOCKS-UNMEASURED` and passes |

Every cache has a test that fills it past its capacity and records its gauge.

The old runtime takes no fixes until the cutover deletes it, so its failures that the matrix has
met are parked (`PARKED_TESTS` and `PARKED_RECORDS` in `scripts/matrix/results.py`, each with its
reason): a parked test's failure, a parked gauge's change between `@100` and `@200`, or a parked
time's cross-GPU outlier is named on the leg's line and fails no leg. A parked gauge's `@drop` still fails. A workload that
failed on the old runtime (`M2V-FAILED harness-old/<row>`, which the harness prints beside its
`FAILED` line) fails no leg either: the budget below names its row as not compared. The list goes with the old
runtime.

The harness's first-use records (`harness-new/<set>/first_use_*`: how many of the new runtime's
first pipeline uses waited, and for how long) depend on the driver's shader cache, which outlives a
run. The two runtimes' rows are timed in one process, `harness --runtime new,old`: each workload
runs on the new runtime, then on the old, so their passes alternate on the GPU, the node and the
driver as they are at that moment, and a clock or a co-runner that moves one moves the other
alike. The new runtime runs first in each workload, so its first uses compile before the old
runtime's of the same workload, as cold as in a process of its own. The process has a fresh empty
cache directory (`MESA_SHADER_CACHE_DIR` for
Mesa, `__GL_SHADER_DISK_CACHE_PATH` for NVIDIA; on the Windows host, `LOCALAPPDATA` and
`USERPROFILE` point into it, which the Intel driver's cache follows), and a fresh empty store for
each runtime's own pipelines (the harness's `--store`: the old runtime's shader cache, the new
runtime's cache directory), so its first uses and its `create_state` row are cold on both
runtimes: those records are judged. A short second run on the same directory and store follows
(`--runs 5 --warmup 1 --no-settle`), and its first-use records are kept as
`harness-new-warm/<set>/first_use_*`, the same first uses with both warm; they are parked.

Both runs skip the creation rows (the harness's `--skip 'create_*'`): those run alone after them,
`--only 'create_*'`, in five processes of their own, each timing both runtimes in the same order
on a fresh empty cache directory and store, so no row before them moves their times: a
creation's first sample on the new runtime is cold, and the old runtime's follows it on the same
cache. The first process's records are kept whole, its devices' gauges at drop as `harness-<runtime>/creates/device/...`; of each later process, only the
creation rows' first samples (`<row>/first`, `/first_call`, `/first_one_thread`, `/first_take`,
`/next_take`, `/first_allocation_total`, `/device_first`, one sample per process), as
`harness-first/<row>/<n>/<runtime>`,
parked for the `time` rule.

The harness budget compares the two runtimes rather than two commits, apart from the `time` rule
and the parked records. The old runtime, frozen, is its noise meter: on each GPU, the old
runtime's time a row is judged on (its GPU time where it is judged on that, a creation row's
median of five processes) is compared with the same time in the leg's base, the previous round,
and a row whose old time moved more than 10% either way is VOID for the round on that GPU: it is
named on the leg's line and judged on nothing, and its comparison comes only from a re-time in
one process with both runtimes interleaved. The report shows that move on every row the base has
(`FROZEN_DRIFT` in `scripts/matrix/results.py`). It measures noise only between rounds of one
harness, one old runtime and one timing shape: where the trees of `crates/metal2vulkan-harness`
or `crates/metal2vulkan-runtime` differ between the two commits, every row but the creation rows
reads `rebased` and voids nothing, and where the base was timed in another shape (the note's
`shape`), every row does. On each GPU, a harness row timed on both runtimes in the leg
(`harness-new/<row>` and `harness-old/<row>`) holds the new runtime's time to at most 1.03 times
the old's, times the row's floor on that GPU where it has one: the widest ratio between two of
the old runtime's own runs of the row there (`scripts/matrix/floors.tsv`, measured again when the
harness or the old runtime's build changes), so a difference within the old runtime's own spread
is not the new runtime's. A row with a GPU time on both runtimes is judged on its GPU time
instead, against the floor of the old runtime's GPU time from the same runs
(`scripts/matrix/gpu-floors.tsv`), its times shown beside it: a single kernel's GPU time spreads
a few percent where its host time spreads up to 1.3 (the XTX's `simd_reduce`: GPU p50 36.96-37.12
µs over 8 runs, host time 44.9-55.8 µs). The frame and burst rows (`WALL_ROWS` in
`scripts/matrix/results.py`), whose GPU time spans the gaps between their command buffers and
spreads as widely, stay on their times and floors, and where both runtimes counted them (Linux,
a node that grants the counter), each must also pass the instructions gate, on the new runtime's
own instructions per timed run: those of the threads it started, which `Gauges::threads` names,
and of the harness's thread (`instructions harness-new/<row>/own_per_run`). A driver's own threads
are left out (`.../others_per_run`, shown beside): they swing by millions of instructions per run
between processes (NVIDIA's on `burst_160`, 0 to 25 million), where the runtime's own do not.
EXACT: the own count is at most 1.03 times its stored count (`scripts/matrix/instructions.tsv`:
one count per row where the legs agree within 3%, else one per driver family). A stored count
only ratchets down: a leg more than 1% under it is named, and nothing raises it without a ruling;
a row with none stored is named. LENIENT, named on every row: the own count is at most the old
runtime's whole count (`.../per_run`), the old runtime naming none of its threads, so its driver's
share cannot be taken out. A new runtime that records its whole count but not its own fails the
row. A time that moves while its count holds names waits. Every row records its counts; only
these are gated on them. A row whose GPU time varies on
one card itself (`CARD_VARIANCE`: the XTX's `sampling` and `constant_managed`) is named on the
leg's line as the card's variance and fails nothing. A row over it fails its leg unless a ruling, made after the work on that
row, gave it a ceiling on that GPU (`HARNESS_CEILINGS` in `scripts/matrix/results.py`); a ceiling
is never taken from a measured ratio, and the greater of a row's ceiling and floor sets its limit.
A listed row fails its leg past its ceiling times 1.03, the noise allowance parity gets, and one
within that is named on the leg's line. A fix lowers its row's ceiling to the ratio it reaches,
and a row within 1.03 leaves the list. A creation row's first sample is judged by the ratio of
the medians of its five processes, the new runtime's over the old's; a first sample alone is
never past its limit. A compute pipeline state row's `/first` is its creation and one dispatch
through to completion, one threadgroup of 64 by 1 by 1, the replay streams' commonest compute
dispatch size (24% of their dispatches), since Metal compiles at the creation and a runtime may
compile at the first dispatch instead, for the dispatch's size, as the old runtime does; its
`/first_call`, the creation call alone, is shown beside it with the ratio of its medians and
never judged, and so is `/first_one_thread`, another content's first dispatched as one thread,
a size of few dispatches (under 4% of them). Each creation row's first sample
is a fresh object's on a device that has gone that way once: the harness makes a warm-up of the
row's kind first (a buffer of the same length, kept; a compute state of another kernel, with its
dispatch; a render state of another fragment function), so what a device pays once, which a guest
that keeps its device pays once in its life, stays out of the judged sample. The warm-up's own
time, `/device_first`, is shown the same way and never judged. A row whose time
either runtime recorded more than once fails, uncompared. A row whose size follows the device
(`long_4cb`) is sized by the first runtime's pass, and the later runs, warm and re-timed ones
included, are given that size (the harness's `--size`); each run records it as
`size harness-<runtime>/<row>`, and a row the two runtimes ran at different sizes fails,
uncompared, its times measuring different work.
A row over its limit is re-timed once in its leg before the leg ends: two pairs of runs of the row
alone, old then new, then new then old (A B B A). It fails only if the median of the pairs' ratios,
of the times it is judged on, is past its limit too. A leg re-times at most 8 rows, the furthest
over first; a row it does not re-time is judged by its one time. The pairs are kept as
`harness-retime/<row>/<pair>/<runtime>` and `harness-retime-gpu/<row>/<pair>/<runtime>` records,
parked for the `time` rule.
Each row's GPU time on each runtime, the p50 over its timed runs of a run's first GPU start to its
last GPU end, is recorded as `time harness-gpu-<runtime>/<row>` where the row has one. The budget
judges a row on it as above, and shows it beside the times of a row it does not judge on it; it is
parked for the `time` rule. `run` and `show` print, under each leg's line, every harness row's time and GPU time
on both runtimes and the ratio of the times, and where the leg counted them, its instructions per
run (with the new runtime's own) and their ratios.
The budget goes with the old runtime.

The old runtime's translations do not move either. The golden hash table compiles through
`Device::new`, the old runtime's path. When its test reports a cell that changed under an unchanged
key, or one under a changed profile fingerprint, the leg fails with `OLD PATH MOVED`. The one
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
- `command-replay play` plays a replay stream ([REPLAY-STREAM.md](REPLAY-STREAM.md)) on a Vulkan
  device through the old runtime and writes its report: the runtime's counters for each group of
  command buffers it waited for, each frame and the whole play, the address words it relocated,
  and a SHA-256 of each checked buffer or texture against the recorded one. `command-replay dump`
  prints a stream's calls.
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
