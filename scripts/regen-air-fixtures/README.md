# regen-air-fixtures

Compiles the owned MSL fixtures in `crates/metal2vulkan-validation/fixtures/public/engine/*.metal` with the
macOS Metal toolchain and writes their companions next to each source. CPU only; nothing runs on
a GPU. `--dir <directory>` does the same for `crates/metal2vulkan-validation/fixtures/public/<directory>/`, which has a
TOOLCHAIN of its own: `reflection/` holds the reflection fixtures and `stage_bound/` the
stage-bound fixture, which the translator's fixture sweeps do not read.

| Output | Produced by | Committed |
|---|---|---|
| `<name>.air` | `xcrun metal -c` (Apple AIR bitcode, typed pointers) | only where the directory already holds one; see "The `.air` companion" below |
| `<name>.ll` | `llvm-dis` of `<name>.air` (opaque-pointer review text) | yes |
| `<name>.air.ll` | `xcrun metal -S -emit-llvm` (typed-pointer text) | yes |
| `TOOLCHAIN` | `xcrun metal --version`, SDK version, LLVM pin, `llvm-dis`/`llvm-as` versions, and each fixture compiled by another Metal toolchain | yes |

```sh
scripts/regen-air-fixtures/regen-air-fixtures.sh            # regenerate every fixture
scripts/regen-air-fixtures/regen-air-fixtures.sh a b        # regenerate fixtures a and b only
scripts/regen-air-fixtures/regen-air-fixtures.sh --check    # fail if a committed companion is stale
scripts/regen-air-fixtures/regen-air-fixtures.sh --dir reflection --check   # the same for reflection/
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/regen-air-fixtures
```

Requires `xcrun` with `metal` and `metal-as`, and `llvm-dis` and `llvm-as` of the pinned LLVM
major (see "The pinned LLVM" below). The unit tests use fake tools and need neither.

A source compiles with the compiler's default language standard unless it has a line
`// metal-std: <standard>` (for example `macos-metal2.2`), which passes `-std=<standard>` to both
compiles. It compiles with the compiler's default fast math unless it has the line
`// metal-fast-math: off`, which passes `-fno-fast-math` to both compiles: the module then records
`air.compile.fast_math_disable`, the plain MSL names reach the precise `air` symbols, and no float
operation carries a fast-math flag, so a lane whose input is a NaN, an infinity, or a signed zero
asks a defined question (under `nnan` or `ninf` such an operand or result is poison, and under
`nsz` either zero is correct). No other value is accepted.

Every source compiles for a stated macOS deployment target: the version of a line
`// metal-macos-min: <version>` (a fixture whose framework headers spell their ABI by deployment
target, such as MetalPerformancePrimitives' 26.2 tensor-op names, or one that needs AIR 2.9's
27.0), else 26.0, the target the other fixtures were compiled for (`default_macos_min` in the
script). Both compiles get `-mmacosx-version-min=<version>`, which outranks
`MACOSX_DEPLOYMENT_TARGET`. Without it the compiler's default follows the host: on macOS 27 it is
27.0, which moves the triple from `air64_v28-apple-macosx26.0.0` to `air64_v29-apple-macosx27.0.0`
along with the AIR and language versions the module records.

A line `// air-rename: <declared> <intrinsic>` (any number of them) names a function the source
declares and never defines (its mangled name, such as `_Z9rfs_yieldt` for `void rfs_yield(ushort)`)
and an `llvm.*` or `air.*` intrinsic MSL has no spelling for (such as `llvm.agx3.yield`). The typed
text is renamed, and the fixture's module is `xcrun metal-as` of it: `<name>.air` is that module and
`<name>.ll` its disassembly. Apple's assembler gives an intrinsic it knows its own declaration
(`llvm.agx3.yield` becomes `convergent noduplicate nounwind`, the parameter's attributes dropped),
so a renamed fixture's module is not compared with the one `xcrun metal` compiled; `llvm-as` of the
pinned major must still accept `<name>.ll`, and `--check` still compares the companions. The call
sites keep the arguments and attributes the compiler wrote for the declaration. A name the module
does not declare, or defines, fails the run.

Sources are compiled from stdin, so no absolute path reaches the output (`source_filename` is
`-` and there is no `air.source_file_name`). Each run checks the typed text: `xcrun metal-as` of
`<name>.air.ll` must disassemble to exactly the text of `<name>.ll`, so the committed typed text
reassembles into the same module that `xcrun metal` compiled. That is the existing `.air.ll`
convention: macOS tests get typed-pointer bitcode from `metal-as` without a committed `.air`, and
tests elsewhere use `llvm-as` of `<name>.ll` for the opaque-pointer path.

Output is deterministic for a given toolchain: two runs are byte-identical, on any host and under
any locale. The script pins its collation to byte order, so the `*.metal` glob it walks and
`TOOLCHAIN`'s sorted fixture lines do not follow the caller's. A different
toolchain may change the companions; `TOOLCHAIN` records which one produced the committed files,
and the change is reviewed like any other fixture edit.

**Mixed Metal toolchains.** `TOOLCHAIN` records a base Metal toolchain (`metal:`, `macosx-sdk:`)
and a line `fixture <name>: metal: ...; macosx-sdk: ...` for each fixture compiled by another
one, in byte order, so adding a fixture after an Xcode update does not mean recompiling every
other fixture:

- A plain run regenerates every fixture with the host's toolchain and records it alone, a
  deliberate re-keying of the whole set.
- A run naming fixtures regenerates only those, records them under the host's Metal toolchain
  (no line when that is the base), and keeps every other record. Every fixture shares one LLVM
  record, so it refuses when the host's `llvm-dis` or `llvm-as` is not the recorded one.
- `--check` regenerates and compares exactly the fixtures recorded under the host's Metal
  toolchain, prints `not checked: <name> (compiled by <toolchain>)` for each other one, and ends
  with the counts of checked and unchecked fixtures. It fails on a stale companion, on a fixture
  whose `.ll` or `.air.ll` records another Metal version (`!llvm.ident`) or SDK (`SDK Version`)
  than `TOOLCHAIN` records for it, on one whose `target triple` names another macOS version than
  its source selects (checked for every fixture, compiled or not), on `TOOLCHAIN` fixture lines in
  any order but byte order, and when the host's LLVM tools differ from the recorded ones.

Fixtures recorded under another Metal toolchain than the host's (those compiled by Metal
32023.883, SDK 26.5, on a host with 32023.921) are listed as not checked; re-keying them is a
separate, deliberate regeneration.

**The pinned LLVM.** The `.ll` companions are written for LLVM 21 (`llvm_major` in the script),
the oldest LLVM whose `llvm-as` the gates assemble them with: Linux CI and the Linux gate image
install Ubuntu 26.04's LLVM 21, while the macOS host may run a newer Homebrew LLVM. A newer
`llvm-dis` may use spellings an older `llvm-as` rejects (LLVM 23 writes `splat (float f0x...)` in
`stage_io.ll`), so:

- `llvm-dis` of exactly that major writes every `.ll`, and `llvm-as` of that major must accept
  each one before anything is written or compared; a rejection fails the run and `--check`.
- Each tool is `METAL2VULKAN_LLVM_DIS` / `METAL2VULKAN_LLVM_AS` when set (refused unless it is
  that major), else the first `llvm-dis-21` / `llvm-as-21` (Debian's versioned names), then
  `llvm-dis` / `llvm-as`, on `PATH` that is that major. A newer LLVM earlier on `PATH` is passed
  over, so the default Homebrew `llvm` and `llvm@21` can both be on `PATH`. With no LLVM 21 tool
  the script fails, naming the versions it found.
- `TOOLCHAIN` records the pin and both tools' versions, and `--check` compares them exactly.

On macOS with Homebrew:

```sh
brew install llvm@21
PATH="$PATH:$(brew --prefix llvm@21)/bin" scripts/regen-air-fixtures/regen-air-fixtures.sh --check
```

Moving the pin to a newer major is a regeneration like any other toolchain change, and is only
sound once every gate's `llvm-as` is at least that major.

**The `.air` companion.** A run refreshes a `<name>.air` the fixture directory already holds and
writes none where there is none. `engine/` holds seven and `reflection/` one, which tests read as
Apple AIR bitcode and a run keeps in step with their text; for every other fixture the `.air` is
an intermediate of the run and stays in its scratch directory.

Putting one beside them is a decision, not a side effect of a run. `xcrun metallib` resolves each
shared `air.*` symbol to one definition for the whole link, and which deployment target that
definition comes from depends on the whole input set, so one more module can make a symbol that
linked before fail with `Cannot link symbol '<air.*>': Used in deployment target <a> but defined
in deployment target <b>`. `metal2vulkan-air`'s `container::tests::committed_air_fixtures_parse_and_link`
links every `.air` the directory holds, so a stray one fails it; `.gitignore` un-ignores this
tree's `*.air`, so `git status` names it, and deleting it is the fix.
