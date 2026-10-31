# Architecture

How the translator turns AIR into Vulkan pipelines: the stages, the rules that hold across them,
and where each piece lives. The consumer's view of the same flow is [API.md](API.md).

## The pipeline

```text
metallib | AIR bitcode
  ── parse ───────► Library: the container's functions                  air::container, library
  ── decode ──────► a typed AIR module and its ABI                      air::{bitstream, bitcode, ir, intrinsics, abi}
  ── specialize ──► one inlined entry: function constants baked,        air::canon
                    constants propagated, linked functions resolved,
                    pointee types erased
  ── analyze ─────► MemoryPlan: provenance, one representation          memory
                    per value, slot windows
  ── select ──────► Plan: a strategy per instruction and interface      lower::select, lower::strategy
                    element, push-constant placement, requirements,
                    the output-size estimate (checked here)
  ── structurize ─► a structured tree over the AIR CFG                  structurize
  ── emit ────────► a typed SPIR-V module through the builder           structurize::emit, lower::*, spirv::builder
  ── tidy ────────► single-store promotion, dead values removed,        structurize::promote, spirv::{dead, roll}
                    repeated regions of large entries rolled into loops
  ── verify ──────► the full verifier; requirements within the profile  spirv::verify, spirv::requirements
  ── serialize ───► SPIR-V 1.5 words with typed relocations             spirv::serialize
  = CompiledFunction, keyed by FunctionCompileKey (the caller may cache it)
  ── link ────────► bindings, Locations, spec ids, the pipeline         pipeline::link
                    layout, pipeline-wide limits, the generated         pipeline::link::geometry
                    geometry stages (barycentric weights read as
                    varyings, a selected slice or viewport stored
                    from the first vertex)
  ── relocate ────► the final words of each stage                       pipeline::relocate
  = ComputePipeline | RenderPipeline | tile and mesh pipelines

BindState ── shape ──► BindingPlan (per pipeline and shape) ── instantiate ──► dispatch or draw commands   bind
```

`pipeline::compile::compile_stage` runs one function from decode to serialize and keeps only the
relocatable module and plain metadata; the decoded AIR, the plan and the typed module are dropped
before it returns. It runs under `catch_unwind`, so a panic in the core becomes
`Error::Internal`, never an unwind into the caller.

Three rules hold across the arrows:

- **Each stage maps valid typed input to valid typed output.** No stage produces a form a later
  stage repairs, and nothing parses SPIR-V words to decide what to emit.
- **Nothing after `select` changes which instructions a function contains**, except the tidy
  passes, which preserve meaning. Link and relocate write only literal numbers (descriptor set and
  binding, `Location`, `SpecId`) into slots the serializer reserved, so one compiled function is
  reused by every pipeline it links into.
- **One representation per feature and device, chosen before emission.** `select` decides every
  strategy from the AIR and the device profile. There are no retries, no candidate modules and no
  validator in the loop. A family that cannot lower an input refuses it.

The library runs no external tool: it reads bitcode with its own reader and verifies with its own
verifier. The command line adds `llvm-as` for `.ll` input and `spirv-val` as an independent check.

## Where each piece lives

The AIR front end is its own crate, `metal2vulkan-air` (`crates/metal2vulkan-air/src/`). The
translator names it `air`, with its `graph` and `budget` modules at the translator's root. The
rest is `crates/metal2vulkan/src/`.

| Module | Role |
|---|---|
| `air` (`metal2vulkan-air`) | The container (metallib, bitcode wrapper, raw module), the bitstream and bitcode readers, the typed IR, the `air.*` and `llvm.*` intrinsic families, the metadata ABI (`AirAbi`), and canonicalization (`canon`: inlining, function constants, SCCP, devirtualization, linking, pruning, pointee erasure, the speculation of the short branch arms Metal's backend runs in every lane, and the split of a whole array's or struct's load or store in buffer or threadgroup memory into its members' accesses). Canonicalization also writes the formats the binder and runtime read: the visible-function table's words and the rasterization-rate map's parameter data |
| `graph` | The shared RPO, dominator tree, loop forest and SCC |
| `budget` | `Budget`, `CancelToken`, and the per-attempt `Meter`: deadline, cancellation, work units, the byte ledger and every structural cap |
| `memory/` | Pointer provenance, the `MemoryPlan`, leaf access emitters, arenas, root and handle lookups, word composition, atomics |
| `lower/` | `select` (routing every instruction, intrinsic and parameter into the `Plan`), `strategy`, `push` (push-constant placement), `estimate` (output size), `specialized` (bound-texture values read as push data), and one module per family: `arith` (each float operation in the AIR's order, decorated by its own fast-math flags, so the device's compiler groups and fuses only what they allow; `half_chain`, a fast-math `half` chain held at `float` precision between its roundings where the device has no 16-bit floats), `access`, `texture`, `sampler`, `simd`, `sync`, `fragment`, `stage_io`, `dispatch`, `tessellation`, `mesh`, `raytracing` |
| `structurize/` | The structurizer over the AIR CFG, the emit walk that drives the families, and single-store promotion |
| `spirv/` | The typed SPIR-V IR and interners, the module and function builders, dead-value removal, the roll pass, the verifier, requirement derivation and the serializer. `spirv::interp` is a CPU interpreter used only by tests |
| `spv.rs` | The SPIR-V grammar's opcodes and enumerants the translator names, generated from Khronos's grammar (`scripts/regen-spirv-grammar/`) |
| `spirv_binary/` | A reader of SPIR-V's binary form over the grammar's instruction table, used only by tests |
| `device/` | `DeviceDescription` (from a Khronos profile, or from `ash`), `DevicePolicy`, and the derived `DeviceProfile` every decision reads |
| `library/` | `Library`, `Function`, `CompiledFunction`, `FunctionCompileKey` |
| `reflection/` | Metal-shaped reflection of functions and pipelines |
| `pipeline/` | Descriptors, compile entry points, link and relocate, `PipelineInterface`, the pipeline layout, vertex input and fixed-function state, runtime facts, precision notes |
| `bind/` | The binder: `BindState` and `BindShape`, `BindingPlan`, per-command instantiation, neutral resources, argument, handle and heap tables, residency, indirect-command replay, and the helper kernels the runtime records. It never calls Vulkan |
| `cache/` | `PipelineKey`, `SEMANTICS_VERSION`, the artifact encoding |
| `metal/`, `vk/`, `ash_interop/` | Metal's enums and format facts; plain-data Vulkan values; `ash` conversions (feature `ash`) |
| `error/` | The typed errors every entry point returns |
| `main.rs` | The command line (feature `cli`) |

The translator core (`lower`, `memory`, `spirv`, `spv`, `structurize`) and every submodule of a
public module are crate-private; no feature or cfg exposes them. The runtime
(`crates/metal2vulkan-runtime`), validation and devtools use the public API only. A test that
needs an internal item is one of the crate's own tests (`src/tests/`).

## Decisions come from structure and the device

- **The device profile.** `Device::new` derives one `DeviceProfile` from the description and the
  policy. Every device-dependent choice (subgroup tier, buffer views, storage widths, robustness,
  push-constant room, limits, measured driver deviations) reads it, and its fingerprint is part of
  every cache key.
- **The device's native forms.** The maths, the grouping and fusion of fast-math arithmetic, NaN
  bits, minimum and maximum, `half` rounding and blends, the sampler and the texel and color
  conversions are the device's own on every device,
  each difference from Metal recorded as a precision note; on MoltenVK on Apple's GPUs most are
  Metal's own. The policy's accuracy switches are the runtime's ([FAST-PATHS.md](FAST-PATHS.md)).
- **Structure, not names.** Families decide from IR types, address spaces, the metadata ABI and the
  `air.*` and `llvm.*` intrinsic contract. No branch keys on a function, type or variable name.
- **Typed refusals.** An input the device cannot run exactly is `Error::Unsupported`, naming the
  Metal feature and what would make it translatable. A missing pipeline input (a function-constant
  value, a stage-input or vertex descriptor, a storage texture's format) is `Error::NeedsInput`,
  asked for and never guessed. A deterministic cap is `Error::LimitExceeded`. Only a deadline
  (`Error::Budget`) or a cancellation (`Error::Cancelled`) is transient.
- **Recorded differences.** Where the output gives a near-equivalent of Metal's answer, the stage
  records a `PrecisionNote`, and `Reconvergence::ImplementationDefined` where the profile lacks
  maximal reconvergence.

## Validity by construction

- **The builder makes invalid modules hard to write.** Typed constructors check operand and
  result types; storage-class rules hold by construction; `if_`, `switch_` and `loop_` are the only
  construct openers, each closed by its scope, so constructs nest correctly.
- **Types are interned with their layout**, so a type is unique and its explicit layout is part of
  its key.
- **Capabilities and extensions are derived** from what the module contains, never written by a
  lowering, and must lie within the profile (else `Error::Internal`, kind
  `RequirementsExceedProfile`).
- **The serializer** writes the logical layout and numbers result ids densely in order, so equal
  modules give equal words. Nothing nondeterministic (such as hash-map order) reaches the output.
- **The verifier always runs**, in release builds too, before any word leaves the crate:
  ownership, SSA dominance, structured control flow and canonical loops, convergent instructions,
  atomic pointers, an acyclic call graph, stage rules, the recorded interface and requirements, and
  the profile's rules. A violation is `Error::Internal` (kind `Verify`): a translator bug, never an
  input error.

## Structurization

The structurizer reads the CFG through a view: blocks are opaque units and terminators are shapes.

1. Prune unreachable blocks and fold trivial branches.
2. Make an irreducible graph reducible with a multi-entry dispatch; refuse an irreducible region
   that holds a convergent instruction, and nesting or output past the structural limits.
3. Find the loop nest.
4. Demote every block-crossing value and phi to a `Function` variable.
5. Canonicalize loops: back edges become `continue`, exits a `break` through one exit dispatch per
   loop.
6. Place every acyclic region on the dominator tree, innermost loop first.
7. Let tail jumps fall out and route the rest through one pending variable.
8. Emit the tree through the builder, each block lowered by the plan's strategies.
9. Promote single-store variables back to values where dominance allows.

The output size is checked against `Limit::OutputBlocks` before anything is emitted.

## Bounded work

Every compile takes a `Budget`: a wall deadline, a `CancelToken`, a deterministic work-unit cap and
a byte ledger. The `Meter` is threaded through every loop that scales with the input: it ticks work
units and charges each input-sized container before reserving it, so a refusal comes before the
allocation and the same input refuses the same way on every host. The crate keeps no global cache
and memoizes nothing; the caller caches compiled functions, pipelines and plans under the public
keys. A consumer that needs a hard memory ceiling or crash isolation compiles in a child process
([HOWTO.md](HOWTO.md#5-bound-a-compile)).

## Checks

Match the evidence to the claim; [VALIDATION.md](VALIDATION.md) has the full ladder.

| Check | Catches |
|---|---|
| clippy with warnings denied | warnings and common defects |
| unit and integration tests | construction and lowering regressions |
| the always-on verifier, and `spirv-val` in tests and the command line | invalid modules |
| interpreter tests (`spirv::interp`) | wrong compute lowering, out-of-bounds access, non-termination |
| the fixture inventory (`tests/public_fixture_expectations_engine.tsv`) | a fixture that stops compiling or starts refusing on a committed profile |
| golden output hashes (`src/tests/golden_hashes.rs`) | unintended output changes |
| authored cases run on Metal and Vulkan | answers that differ from Metal's |
| bounded release-mode corpus translation | time, memory and coverage over real AIR |

A valid module is necessary, not sufficient. A change that only silences the verifier or
`spirv-val` is unsound; unknown semantics stay refusals.
