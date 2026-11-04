# Contributing

How to build, test and change metal2vulkan. [AGENTS.md](AGENTS.md) holds the project's rules;
they bind people too.

## Prerequisites

- rustup. `rust-toolchain.toml` pins the nightly this tree builds with (rustup installs it on first
  use); the published crates promise the stable `rust-version` in their manifests (1.98).
- [cargo-nextest](https://nexte.st): `cargo install cargo-nextest --locked`.
- The external tools the tests and validation use (the libraries run none):
  - `llvm-as` (LLVM 21 or newer) for the `.ll` fixtures, and `llvm-dis` for validation's harvest
    and fixture tooling;
  - `spirv-val` and `spirv-as` from SPIRV-Tools;
  - `xcrun metal-as` on macOS, for the `.air.ll` fixtures (tests that need it skip elsewhere);
  - `spirv-cross`, for the checks of the MSL MoltenVK would write. Without it they print `SKIPPED`;
    `METAL2VULKAN_TEST_REQUIRE_SPIRV_CROSS=1` makes its absence a failure.

  Each is found on `PATH` or through `METAL2VULKAN_<TOOL>` (`METAL2VULKAN_LLVM_AS`,
  `METAL2VULKAN_SPIRV_VAL`, ...). With Homebrew on macOS, LLVM is keg-only:
  `PATH=/opt/homebrew/opt/llvm/bin:$PATH`.
- A Vulkan driver or a Mac with Metal only for the device tests and authored-case runs
  ([docs/VALIDATION.md](docs/VALIDATION.md)).

## Layout

```text
.
├── crates/
│   ├── metal2vulkan/             # the translator library and the CLI (feature `cli`)
│   │   ├── src/                  #   src/tests/: tests that reach internal items
│   │   ├── tests/                #   integration tests through the public API, golden tables
│   │   └── examples/             #   api_preview.rs, which docs/API.md quotes
│   ├── metal2vulkan-air/         # the AIR front end
│   ├── metal2vulkan-runtime/     # the Metal API on Vulkan (README, examples)
│   ├── metal2vulkan-validation/  # unpublished: cases, executors, corpus tools
│   │   ├── fixtures/public/      #   our own synthetic fixtures (.ll, .air.ll, .metal, .spvasm)
│   │   ├── profiles/             #   committed Khronos device profiles
│   │   └── corpus/               #   authored cases and observations (local/ is private, ignored)
│   └── metal2vulkan-devtools/    # unpublished: tool runner, fixture assemblers
├── docs/                         # guides and generated references
├── scripts/                      # developer utilities and generators
├── Cargo.toml                    # the virtual workspace
├── LICENSE                       # the notice: GPL-2.0-or-later OR LGPL-3.0-or-later
└── LICENSE-*                     # the texts it names: GPL v2, LGPL v3, and GPL v3
```

## The loop

```sh
cargo fmt --all
cargo clippy --workspace --all-targets --all-features -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --all-features --no-deps
scripts/api-docs   # the translator's API reference; fails on a link to a hidden item
cargo nextest run -p metal2vulkan -p metal2vulkan-air -p metal2vulkan-runtime -p metal2vulkan-devtools --all-features
cargo nextest run -p metal2vulkan-validation --all-features
cargo test --workspace --all-features --doc
```

None of these opens a device. Device tests compile only under `--cfg m2v_gpu` or
`--cfg m2v_cpu_icd` ([docs/VALIDATION.md](docs/VALIDATION.md)).

To build the command:

```sh
cargo build --release -p metal2vulkan --features cli
```

A debugger session that needs variables builds with `--profile debugger` (full debug info for the
workspace and its dependencies). Dev builds keep only line tables.

### Many trees on one machine

Parallel worktrees and one-shot builds write a lot to disk. Three scripts keep that down:

- `scripts/lean-tree worktree PATH BRANCH START` adds a full worktree whose
  `crates/metal2vulkan-validation/corpus/observations` (1.5 GiB of the 1.6 GiB tracked tree) are
  copy-on-write clones of the main worktree's; git rewrites only the files that differ at `START`.
  `scripts/lean-tree snapshot REV DIR` extracts a tree without the observations (pass
  `--with-observations` for one that runs validation).
- `scripts/seed-target new TARGET SEED WORKTREE` starts a target directory as a copy-on-write clone
  of a seed, so only the workspace's crates compile. `scripts/seed-target save TARGET SEED WORKTREE`
  makes the seed from an idle warm target. Keep one seed per set of `RUSTFLAGS`.
- `scripts/xctrace-run TRACE ARGS...` records with Instruments and removes the raw
  `instruments*.ktrace` files a recording leaves in the temp dir, an aborted one included.

Builds are incremental. Every compile goes through `scripts/rustc-wrapper`, which hands it to
[kache](https://github.com/kunobi-ninja/kache) when kache is on `PATH` (outside CI) and to rustc
otherwise; kache restores matching compiles from its local store and imports incremental state
between worktrees. `RUSTC_WRAPPER=` (empty) builds without it.

## Where tests go

- A test that names an internal item is a unit test of its crate: under `src/tests/` (or its
  module's own `tests`), naming items through `crate::`. Run one file with
  `cargo nextest run -p metal2vulkan --all-features --lib -E 'test(/^tests::<file>::/)'`.
- `crates/metal2vulkan/tests/` holds integration tests through the public API only.
- `crates/metal2vulkan/tests/api_surface.rs` pins every public name of the translator. A new public
  item gets its row there.
- Fixtures are synthetic and ours, under `crates/metal2vulkan-validation/fixtures/public/`. A new
  fixture's rows in `crates/metal2vulkan/tests/public_fixture_expectations_engine.tsv` are blessed
  from the test rather than written by hand, with
  `cargo nextest run -p metal2vulkan --all-features --run-ignored ignored-only -E 'test(bless_engine_inventory)'`
  (on macOS, which has every assembler); the parts name every moved cell and every new or gone row
  first, so read those before blessing. Its golden printout under `tests/air_input/` is blessed with
  `cargo test -p metal2vulkan --all-features --lib tests::air_input -- --ignored`.
- The manifests set `autobins = false`: a new binary needs its own `[[bin]]` table.

## Golden output hashes and `SEMANTICS_VERSION`

`src/tests/golden_hashes.rs` pins a hash of the final SPIR-V (or the refusal class) of every public
fixture entry point on every committed profile, in `crates/metal2vulkan/tests/golden_hashes.tsv`.
`cache::SEMANTICS_VERSION` is part of every cache key, so a change that moves a cell while its
fixture and profile are unchanged must bump `SEMANTICS_VERSION`
(`crates/metal2vulkan/src/cache/mod.rs`) in the same change. New, removed or edited fixtures and
profiles need only the bless. The table records cells only, no version: a `SEMANTICS_VERSION` bump
alone leaves it unchanged, and the bless on `development` refuses a cell that moved under an
unchanged fixture and profile when `SEMANTICS_VERSION` is still its value at the table's last bless.

The tables are re-blessed on `development` after merges, so a change need not carry its own bless.
One command blesses both (on macOS, which has `xcrun metal-as`):

```sh
PATH=/opt/homebrew/opt/llvm/bin:$PATH cargo nextest run -p metal2vulkan --all-features \
  --run-ignored ignored-only -E 'test(bless_golden_hashes)'
```

Never merge a table by hand: after a merge take `development`'s tables and bless again. The other
versions move the same way when their behavior changes: `ARTIFACT_FORMAT_VERSION` (the artifact
encoding) and `RUNTIME_VERSION` (what the runtime does with an unchanged artifact).

## Generated docs

Edit the source, then regenerate:

| Doc | Generator | Source |
|---|---|---|
| `docs/REFUSALS.md` | `python3 scripts/refusal-surface/refusal_surface.py` (`--check` to verify, `--missing-basis` and `--missing-reach` to list the gaps, `--selftest` for the basis parse's own cases) | the code's refusal types, the `Basis:` line each refusal's doc comment ends with; `reach.tsv` beside the script |
| `docs/METAL-COVERAGE.md` | `python3 scripts/metal-coverage/coverage.py` (`--check` to verify) | `scripts/metal-coverage/coverage.tsv` |
| `docs/CONFORMANCE-COVERAGE.md` | `python3 scripts/conformance-coverage/coverage.py` (`--check` to verify, `--selftest` for the option-space axis's own tests) | the suite's sources, its committed recordings, `scripts/metal-coverage/coverage.tsv`, `objc2_metal.tsv`, and the generated Metal enum tables |
| `crates/metal2vulkan/src/spv.rs`, `crates/metal2vulkan/src/spirv_binary/table.rs` | `scripts/regen-spirv-grammar/regen-spirv-grammar.sh` | Khronos SPIRV-Headers' grammar JSON, and `items.txt` beside the script |
| `crates/metal2vulkan-validation/profiles/*.json` | `scripts/regen-device-profiles/` | gpuinfo reports and local captures |

`crates/metal2vulkan/tests/api_signatures.tsv` is written the same way: the signature-level public
surface of the three published crates, by `python3 scripts/api-signatures/signatures.py --bless`
(`--check` to verify, `--selftest` for the derivation's own tests), from those crates' sources
alone. `crates/metal2vulkan/tests/api_signatures.rs` says what it pins, what it cannot see, and
what blessing it asserts; a removed or changed line is a breaking change of a published crate.

Four docs are checked by tests. `docs/REFUSALS.md` must be the surface the sources state, and
every variant of the runtime's `Problem` must end its doc comment with a `Basis:` line that says
where the belief that Metal refuses that input comes from and cites its evidence
(`crates/metal2vulkan/tests/refusal_surface.rs`, which says what it cannot see; a basis that is
not settled is `Basis: unknown (why)`, never nothing). `docs/FAST-PATHS.md` must name every
`DevicePolicy` accuracy choice and the notes it records (`tests::fast_paths`); `docs/API.md` must list every degrade code
with its category and quote `examples/api_preview.rs` verbatim in its `rust` blocks; and
`docs/METAL-COVERAGE.md` must carry the coverage table's totals.

## Changes

- Follow [AGENTS.md](AGENTS.md): structure over names, honest refusals, fast defaults with exact
  switches (and their row in `docs/FAST-PATHS.md`), tests for every fixed bug class.
- One concern per commit when practical; run `cargo fmt --all` first.
- Update the docs a consumer reads when what they describe changes: `docs/API.md` and
  `examples/api_preview.rs` for the translator's API, the runtime's README and rustdoc for the
  runtime, `docs/HOWTO.md` for integration, `docs/ARCHITECTURE.md` for the translator's stages,
  `README.md` for the command line.
- `CHANGELOG.md` lists releases in a few lines each; add to the top, unreleased version.

## License

metal2vulkan is dual-licensed, `GPL-2.0-or-later OR LGPL-3.0-or-later`, at the recipient's choice
([LICENSE](LICENSE)). Contributions are accepted under the same dual license: a change you submit
can be redistributed and modified under either.

The license files sit at the root and, because cargo packages each crate from its own directory,
as copies in every published crate (`crates/metal2vulkan`, `crates/metal2vulkan-air` and
`crates/metal2vulkan-runtime`). Change them at the root and copy them over; a validation test (`published_crates`) holds every copy equal to the root's file
and the notice's SPDX line equal to the manifests' `license`.
