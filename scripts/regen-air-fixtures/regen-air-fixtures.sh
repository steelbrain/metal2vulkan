#!/usr/bin/env bash
# Regenerates the AIR companions of the owned MSL fixtures in
# crates/metal2vulkan-validation/fixtures/public/engine/, or in another directory under crates/metal2vulkan-validation/fixtures/public/
# (`--dir`) (CPU only; never touches a GPU).
#
# For every <name>.metal it writes, next to the source:
#   <name>.air     Apple AIR bitcode from `xcrun metal`, written only where the directory
#                  already holds one (see "The .air companion" below)
#   <name>.ll      review companion: `llvm-dis` of <name>.air (opaque-pointer text)
#   <name>.air.ll  typed-pointer text from `xcrun metal -S -emit-llvm`; `xcrun metal-as`
#                  reassembles it into the same module, so macOS tests get typed-pointer
#                  bitcode without a committed .air
# and records the toolchain in TOOLCHAIN. A source line `// metal-std: <standard>` passes
# `-std=<standard>` to both compiles, a line `// metal-fast-math: off` passes `-fno-fast-math`
# (the module then records `air.compile.fast_math_disable` and its float operations carry no
# fast-math flags), and a line `// metal-macos-min: <version>` passes
# `-mmacosx-version-min=<version>` (a fixture whose framework headers spell their ABI by deployment
# target, such as MetalPerformancePrimitives' 26.2 tensor-op names); a source without one compiles
# for the pinned deployment target (default_macos_min below). A line
# `// air-rename: <declared> <intrinsic>` renames a function the source declares and never defines
# to an `llvm.*` or `air.*` intrinsic MSL cannot spell (such as `llvm.agx3.yield`) in the typed
# text; <name>.air is then metal-as of the renamed typed text, and <name>.ll its disassembly. A line
# `// air-by-value-buffer: <arg>` rewrites the node of the read-only `constant` buffer argument
# named <arg> to the one Apple's compiler writes for a by-value argument it passes as a constant
# pointer, which MSL cannot spell: a read-only `device` buffer of one `char`; the module is then
# metal-as of the rewritten typed text, as for a rename.
#
# The .air companion: a run refreshes a <name>.air the fixture directory already holds, and
# writes none where there is none. The committed outputs are <name>.ll, <name>.air.ll and
# TOOLCHAIN; a handful of fixtures also commit the Apple-compiled <name>.air that tests read as
# bitcode, and those stay in step with their text. For every other fixture the .air is an
# intermediate and stays in the scratch directory, which the run removes. An .air beside the
# committed ones is not a harmless leftover: `xcrun metallib` resolves each shared `air.*` symbol
# to one definition for the whole link, and which deployment target that definition comes from
# depends on the whole input set, so one more module can make a symbol that linked before fail
# with `Cannot link symbol '<air.*>': Used in deployment target <a> but defined in deployment
# target <b>`. An .air an older run left behind is untracked (.gitignore un-ignores this tree's
# *.air), so `git status` names it; delete it.
#
# Sources are compiled from stdin so no absolute path reaches the output. Every fixture compiled is
# checked: metal-as of <name>.air.ll must disassemble to exactly <name>.ll, and llvm-as of the
# pinned LLVM (llvm_major below) must accept <name>.ll.
#
# TOOLCHAIN records a base Metal toolchain (`metal:`, `macosx-sdk:`), the LLVM pin and tools every
# fixture shares, and a line `fixture <name>: metal: ...; macosx-sdk: ...` for each fixture
# compiled by another Metal toolchain. A plain run regenerates every fixture with the host's
# toolchain and records it alone. A run naming fixtures regenerates only those and records them
# under the host's toolchain, keeping every other fixture's record; it needs the LLVM tools
# TOOLCHAIN records.
#
# Tools: xcrun (metal, metal-as), and llvm-dis and llvm-as of the pinned LLVM major. Each LLVM
# tool is METAL2VULKAN_LLVM_DIS / METAL2VULKAN_LLVM_AS when set (it must be that major), else the
# first llvm-dis-<major> / llvm-as-<major>, then llvm-dis / llvm-as, on PATH that is that major,
# so a newer LLVM earlier on PATH is passed over.
# Usage: scripts/regen-air-fixtures/regen-air-fixtures.sh [--dir <directory>] [--check | <name>...]
#   --dir    the fixtures of crates/metal2vulkan-validation/fixtures/public/<directory> (default engine), each
#            directory with its own TOOLCHAIN: the reflection fixtures live in `reflection`, which
#            the translator's fixture sweeps do not read.
#   --check  regenerate into a scratch directory the fixtures TOOLCHAIN records under the host's
#            Metal toolchain and fail if a committed companion differs, naming each other fixture
#            "not checked"; fail also when a fixture's module records another Metal version or
#            SDK version than TOOLCHAIN records for it, or targets another macOS version than its
#            source selects, or when the host's LLVM tools differ.
#   <name>   regenerate only these fixtures (`<name>.metal`).
#
# TOOLCHAIN's fixture lines are sorted by byte value, and the run's own order (the source glob,
# which is the compile order and the order --check reports in) is byte order too, so two hosts
# write the same file whatever their locales collate to.
set -euo pipefail

# Byte order, not the caller's: the `*.metal` glob and the `sort` of the fixture lines both follow
# the collation, and en_US.UTF-8 orders `color_groups_fetch` before `color_groups` where byte order
# puts them the other way round, so without this the committed TOOLCHAIN depends on the host's
# locale (82 of engine's 344 fixture lines move between those two). Character ranges such as
# `[a-z_]` are byte ranges under it too. LC_ALL outranks LC_COLLATE, so it goes first.
unset LC_ALL
export LC_COLLATE=C

# The LLVM major the .ll companions are written for: the oldest LLVM whose llvm-as the gates
# assemble them with (Ubuntu 26.04's LLVM 21 in Linux CI and the Linux gate image). A newer
# llvm-dis may use spellings an older llvm-as rejects (LLVM 23 writes `splat (float f0x...)`),
# so llvm-dis of exactly this major writes each .ll and llvm-as of this major must accept it.
llvm_major=21

# The deployment target of a fixture whose source has no `// metal-macos-min:` line: the one the
# committed companions were compiled for. Without the flag the compiler takes
# MACOSX_DEPLOYMENT_TARGET or a default that follows the host (27.0 on a macOS 27 host), which
# moves the triple (`air64_v29-apple-macosx27.0.0` for `air64_v28-apple-macosx26.0.0`) and the AIR
# and language versions the module records; the flag outranks the variable.
default_macos_min=26.0

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
usage() { echo "usage: $0 [--dir <directory>] [--check | <name>...]" >&2; exit 2; }
subdir=engine
if [[ "${1:-}" == --dir ]]; then
  [[ $# -ge 2 && "$2" =~ ^[a-z_]+$ ]] || usage
  subdir="$2"
  shift 2
fi
dir="$repo/crates/metal2vulkan-validation/fixtures/public/$subdir"
[[ -d "$dir" ]] || { echo "no fixture directory crates/metal2vulkan-validation/fixtures/public/$subdir" >&2; exit 1; }
check=0
names=()
case "${1:-}" in
  --check) check=1; shift; [[ $# -eq 0 ]] || usage ;;
  -*) usage ;;
esac
names=("$@")
for name in ${names[@]+"${names[@]}"}; do
  [[ "$name" =~ ^[A-Za-z0-9_]+$ ]] || usage
  [[ -f "$dir/$name.metal" ]] || { echo "no fixture $name: $dir/$name.metal does not exist" >&2; exit 1; }
done

# Prints the version line of an LLVM tool (e.g. `Homebrew LLVM version 21.1.8`), or nothing.
llvm_version() {
  local text
  text="$("$1" --version 2>/dev/null)" || return 0
  awk 'tolower($0) ~ /version/ { sub(/^ +/, ""); print; exit }' <<<"$text"
}

# Prints the major version in a version line, or nothing.
llvm_version_major() {
  awk 'match($0, /[Vv]ersion [0-9]+/) { print substr($0, RSTART + 8, RLENGTH - 8); exit }' \
    <<<"$1"
}

# Prints the path of LLVM tool $1 of major $llvm_major: the override in the variable named $2
# when set, else the first $1-$llvm_major, then $1, on PATH of that major. Fails naming what it
# found otherwise.
resolve_llvm_tool() {
  local tool="$1" variable="$2" override="${!2:-}" version seen="" name entry candidate
  local entries=()
  if [[ -n "$override" ]]; then
    if [[ ! -f "$override" || ! -x "$override" ]]; then
      echo "$variable=$override is not an executable file" >&2
      return 1
    fi
    version="$(llvm_version "$override")"
    if [[ "$(llvm_version_major "$version")" != "$llvm_major" ]]; then
      echo "$variable=$override is ${version:-of unknown version}, but the fixtures pin" \
        "LLVM $llvm_major" >&2
      return 1
    fi
    echo "$override"
    return 0
  fi
  IFS=: read -r -a entries <<<"$PATH"
  for name in "$tool-$llvm_major" "$tool"; do
    for entry in ${entries[@]+"${entries[@]}"}; do
      candidate="${entry:-.}/$name"
      [[ -f "$candidate" && -x "$candidate" ]] || continue
      version="$(llvm_version "$candidate")"
      if [[ "$(llvm_version_major "$version")" == "$llvm_major" ]]; then
        echo "$candidate"
        return 0
      fi
      seen="$seen"$'\n'"  $candidate: ${version:-unknown version}"
    done
  done
  echo "no $tool of LLVM $llvm_major (the fixtures' pinned LLVM) on PATH${seen:+; found:$seen}" >&2
  echo "put LLVM $llvm_major's bin directory on PATH or set $variable" >&2
  return 1
}

llvm_dis="$(resolve_llvm_tool llvm-dis METAL2VULKAN_LLVM_DIS)" || exit 1
llvm_as="$(resolve_llvm_tool llvm-as METAL2VULKAN_LLVM_AS)" || exit 1
if ! xcrun -sdk macosx -f metal >/dev/null 2>&1 || ! xcrun -sdk macosx -f metal-as >/dev/null 2>&1; then
  echo "xcrun metal/metal-as not found: this script needs the macOS Metal toolchain" >&2
  exit 1
fi

work="$(mktemp -d "${TMPDIR:-/tmp}/m2v-air-fixtures.XXXXXX")"
trap 'rm -rf "$work"' EXIT
out="$work/out"
mkdir -p "$out"

header() {
  printf '; Generated by scripts/regen-air-fixtures from %s.metal (%s). Do not edit.\n' "$1" "$2"
}

shopt -s nullglob
sources=("$dir"/*.metal)
if [[ ${#sources[@]} -eq 0 ]]; then
  echo "no .metal sources in $dir" >&2
  exit 1
fi

# Sets `flags` to the compile flags the source $1's directives select, `renames` to its
# `air-rename` pairs (declared, intrinsic, ...), `by_value` to its `air-by-value-buffer` argument
# names, and `macos_min` to the deployment target it compiles for (`macos_min_from` saying what
# selected it), or fails naming a bad directive.
directive_flags() {
  local source="$1" name std fast_math from to rest arg
  name="$(basename "$source" .metal)"
  flags=()
  renames=()
  by_value=()
  # Optional `// air-by-value-buffer: <arg>` lines: the read-only `constant` buffer argument named
  # <arg> takes the node Apple's compiler writes for a by-value argument it passes as a constant
  # pointer (rewrite_buffer_nodes).
  while read -r arg rest; do
    if [[ ! "$arg" =~ ^[A-Za-z_][A-Za-z0-9_]*$ || -n "$rest" ]]; then
      echo "$name.metal: air-by-value-buffer needs one argument name, not: $arg $rest" >&2
      exit 1
    fi
    by_value+=("$arg")
  done < <(sed -n 's|^// air-by-value-buffer: *||p' "$source")
  # Optional `// air-rename: <declared> <intrinsic>` lines: the source declares a function it never
  # defines (a mangled C++ name), and both companions call the intrinsic MSL has no spelling for
  # instead, as the call sites, arguments and attributes the compiler wrote for the declaration.
  while read -r from to rest; do
    if [[ ! "$from" =~ ^[A-Za-z_][A-Za-z0-9_]*$ || ! "$to" =~ ^(llvm|air)\.[A-Za-z0-9_.]+$ ||
          -n "$rest" ]]; then
      echo "$name.metal: air-rename needs <declared> <llvm.* or air.* name>, not: $from $to $rest" >&2
      exit 1
    fi
    renames+=("$from" "$to")
  done < <(sed -n 's|^// air-rename: *||p' "$source")
  # An optional `// metal-std: <standard>` line selects the language standard (a fixture that
  # needs, e.g., the pre-2.3 discard semantics); otherwise the compiler's default applies.
  std="$(sed -n 's|^// metal-std: *\([A-Za-z0-9.-]*\) *$|\1|p' "$source" | head -n 1)"
  if [[ -n "$std" ]]; then flags=("-std=$std"); fi
  # An optional `// metal-fast-math: off` line compiles without fast math (a fixture whose
  # NaN, infinity, or signed-zero lanes mean something only when no operation carries `nnan`,
  # `ninf`, or `nsz`); the compiler's default is fast math on, and no other value is accepted.
  fast_math="$(sed -n 's|^// metal-fast-math: *\([A-Za-z0-9.-]*\) *$|\1|p' "$source" | head -n 1)"
  case "$fast_math" in
    "") ;;
    off) flags+=("-fno-fast-math") ;;
    *) echo "$name.metal: metal-fast-math must be off, not $fast_math" >&2; exit 1 ;;
  esac
  # An optional `// metal-macos-min: <version>` line sets the deployment target, a dotted
  # version such as 26.2; the pinned default_macos_min applies otherwise.
  macos_min="$(sed -n 's|^// metal-macos-min: *\([A-Za-z0-9.-]*\) *$|\1|p' "$source" | head -n 1)"
  if [[ -n "$macos_min" ]]; then
    if [[ ! "$macos_min" =~ ^[0-9]+(\.[0-9]+)*$ ]]; then
      echo "$name.metal: metal-macos-min must be a dotted version, not $macos_min" >&2; exit 1
    fi
    macos_min_from="its metal-macos-min line"
  else
    macos_min="$default_macos_min"
    macos_min_from="the pinned default"
  fi
  flags+=("-mmacosx-version-min=$macos_min")
}

# Every source's directives are checked before anything is compiled, whichever fixtures a run
# compiles.
for source in "${sources[@]}"; do
  directive_flags "$source"
done

# Applies the `renames` of fixture $1 to module text $2 in place: each declared function's every
# reference `@<declared>(` becomes `@<intrinsic>(`. Fails when the text defines the function or
# never names it.
rename_text() {
  local name="$1" text="$2" i from to
  for ((i = 0; i < ${#renames[@]}; i += 2)); do
    from="${renames[i]}"
    to="${renames[i + 1]}"
    if ! grep -q "^declare .*@$from(" "$text"; then
      echo "$name.metal: air-rename: the module declares no $from" >&2
      exit 1
    fi
    if grep -q "^define .*@$from(" "$text"; then
      echo "$name.metal: air-rename: the module defines $from" >&2
      exit 1
    fi
    sed "s/@$from(/@$to(/g" "$text" >"$text.renamed"
    mv "$text.renamed" "$text"
  done
}

# Applies the `by_value` arguments of fixture $1 to typed module text $2 in place: the one
# `air.buffer` node naming each argument (`!"air.arg_name", !"<arg>"`), a read-only `constant`
# buffer (`air.read`, `air.address_space 2`, its struct description, size, alignment and type name),
# becomes the node Apple's compiler writes for a by-value argument passed as a constant pointer:
# `air.read`, `air.address_space 1`, one `char` (`air.arg_type_size 1`, `air.arg_type_align_size 1`,
# `air.arg_type_name "char"`). The parameter stays a `constant` pointer. Fails when no node or more
# than one names the argument, or the node is not such a buffer's.
rewrite_buffer_nodes() {
  local name="$1" text="$2" arg node count line
  local constant='!"air\.read", !"air\.address_space", i32 2, (!"air\.struct_type_info", ![0-9]+, )?!"air\.arg_type_size", i32 [0-9]+, !"air\.arg_type_align_size", i32 [0-9]+, !"air\.arg_type_name", !"[^"]*"'
  local by_value_node='!"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"char"'
  for arg in ${by_value[@]+"${by_value[@]}"}; do
    node="!\"air\\.buffer\".*!\"air\\.arg_name\", !\"$arg\"[,}]"
    count="$(grep -c -E "$node" "$text" || true)"
    if [[ "$count" -ne 1 ]]; then
      echo "$name.metal: air-by-value-buffer: $count buffer nodes name $arg, not one" >&2
      exit 1
    fi
    line="$(grep -E "$node" "$text")"
    if ! grep -q -E "$constant" <<<"$line"; then
      echo "$name.metal: air-by-value-buffer: $arg is no read-only constant buffer" >&2
      exit 1
    fi
    sed -E "/$node/s/$constant/$by_value_node/" "$text" >"$text.rewritten"
    mv "$text.rewritten" "$text"
  done
}

# Compiles fixture $1 into $out (<name>.ll, <name>.air.ll, <name>.air), checking both companions.
compile_fixture() {
  local name="$1" source="$dir/$1.metal"
  directive_flags "$source"
  xcrun -sdk macosx metal ${flags[@]+"${flags[@]}"} -c -x metal - -o "$work/$name.air" <"$source"
  xcrun -sdk macosx metal ${flags[@]+"${flags[@]}"} -S -emit-llvm -x metal - \
    -o "$work/$name.typed.ll" <"$source"
  "$llvm_dis" -o "$work/$name.dis.ll" <"$work/$name.air"
  # A renamed fixture's module is `xcrun metal-as` of the renamed typed text. Apple's assembler
  # gives an intrinsic it knows its own declaration (`llvm.agx3.yield` becomes `convergent
  # noduplicate nounwind`, without the parameter's attributes), so that module is not the one
  # `xcrun metal` compiled with the plain declaration, and is not compared with it. A fixture whose
  # buffer nodes are rewritten (`air-by-value-buffer`) is likewise metal-as of the rewritten text.
  if [[ ${#renames[@]} -gt 0 ]]; then
    rename_text "$name" "$work/$name.typed.ll"
  fi
  if [[ ${#by_value[@]} -gt 0 ]]; then
    rewrite_buffer_nodes "$name" "$work/$name.typed.ll"
  fi
  xcrun -sdk macosx metal-as "$work/$name.typed.ll" -o "$work/$name.rt.air"
  "$llvm_dis" -o "$work/$name.rt.ll" <"$work/$name.rt.air"
  if [[ ${#renames[@]} -gt 0 || ${#by_value[@]} -gt 0 ]]; then
    cp "$work/$name.rt.air" "$work/$name.air"
    cp "$work/$name.rt.ll" "$work/$name.dis.ll"
  fi
  if ! cmp -s "$work/$name.dis.ll" "$work/$name.rt.ll"; then
    echo "$name: metal-as of the typed text does not reproduce the compiled module" >&2
    diff -u "$work/$name.dis.ll" "$work/$name.rt.ll" >&2 || true
    exit 1
  fi
  { header "$name" "llvm-dis of $name.air"; cat "$work/$name.dis.ll"; } >"$out/$name.ll"
  if ! "$llvm_as" -o /dev/null "$out/$name.ll" 2>"$work/$name.as.err"; then
    echo "$name.ll: llvm-as of LLVM $llvm_major rejects the text:" >&2
    cat "$work/$name.as.err" >&2
    exit 1
  fi
  { header "$name" "typed pointers; xcrun metal-as reassembles it"; cat "$work/$name.typed.ll"; } \
    >"$out/$name.air.ll"
  # Only a fixture the directory already holds an .air for keeps one (see "The .air companion").
  if [[ -f "$dir/$name.air" ]]; then
    cp "$work/$name.air" "$out/$name.air"
  fi
}

# The Metal toolchain of this host, as TOOLCHAIN records one: `metal: <version line>;
# macosx-sdk: <version> (<build>)`.
host_metal="$(xcrun -sdk macosx metal --version 2>&1 | head -n 1)"
host_sdk="$(xcrun -sdk macosx --show-sdk-version) ($(xcrun -sdk macosx --show-sdk-build-version))"
host_toolchain="metal: $host_metal; macosx-sdk: $host_sdk"

# The TOOLCHAIN lines every fixture shares: the LLVM pin and the host's LLVM tools.
llvm_lines() {
  echo "llvm: $llvm_major (pinned: llvm-dis writes each .ll, and llvm-as must accept it)"
  echo "llvm-dis: $(llvm_version "$llvm_dis")"
  echo "llvm-as: $(llvm_version "$llvm_as")"
}

# Writes a TOOLCHAIN with base Metal version line $1 and SDK $2, and the fixture lines in file $3,
# in byte order.
write_toolchain() {
  echo "# Toolchain that produced the .ll/.air.ll companions (scripts/regen-air-fixtures)."
  echo "metal: $1"
  echo "macosx-sdk: $2"
  llvm_lines
  if [[ -s "$3" ]]; then
    echo "# Fixtures compiled by another Metal toolchain (regen-air-fixtures.sh <name>...)."
    sort "$3"
  fi
}

committed="$dir/TOOLCHAIN"

# The value of TOOLCHAIN field $1 (`metal`, `macosx-sdk`).
toolchain_field() {
  sed -n "s/^$1: //p" "$committed" | head -n 1
}

# The fixture line TOOLCHAIN has for fixture $1, without its `fixture <name>: ` prefix, or nothing.
fixture_record() {
  awk -v p="fixture $1: " 'index($0, p) == 1 { print substr($0, length(p) + 1); exit }' \
    "$committed"
}

# Prints `<Metal version line>|<SDK version>` as module text $1 records them (`!llvm.ident` and
# the `SDK Version` module flag), each empty when absent.
module_toolchain() {
  awk '
    /^!llvm\.ident = !\{![0-9]+\}$/ {
      id = $0; sub(/^!llvm\.ident = !\{/, "", id); sub(/\}$/, "", id)
    }
    index($0, "!\"SDK Version\", [") {
      v = $0; sub(/.*!"SDK Version", \[[0-9]+ x i32\] \[/, "", v); sub(/\]\}$/, "", v)
      gsub(/i32 /, "", v); gsub(/, /, ".", v); sdk = v
    }
    { line[NR] = $0 }
    END {
      ident = ""
      if (id != "") {
        for (i = 1; i <= NR; i++) {
          if (index(line[i], id " = !{!\"") == 1) {
            ident = substr(line[i], length(id) + 8); sub(/"\}$/, "", ident); break
          }
        }
      }
      print ident "|" sdk
    }' "$1"
}

# Prints the macOS version of the `target triple` in module text $1 (`26.0.0` for
# `air64_v28-apple-macosx26.0.0`), or nothing when it records none.
module_macos() {
  sed -n 's/^target triple = "[^"]*-apple-macosx\([0-9.]*\)"$/\1/p' "$1" | head -n 1
}

# Whether dotted versions $1 and $2 are equal, a missing component reading 0 (`27.0` is `27`).
same_version() {
  awk -v a="$1" -v b="$2" 'BEGIN {
    na = split(a, x, "."); nb = split(b, y, "."); n = na > nb ? na : nb
    for (i = 1; i <= n; i++) if (x[i] + 0 != y[i] + 0) exit 1
    exit 0
  }'
}

if [[ $check -eq 1 ]]; then
  if [[ ! -f "$committed" ]]; then
    echo "missing: TOOLCHAIN" >&2
    exit 1
  fi
  base="metal: $(toolchain_field metal); macosx-sdk: $(toolchain_field macosx-sdk)"
  status=0
  checked=0
  unchecked=0
  for source in "${sources[@]}"; do
    name="$(basename "$source" .metal)"
    record="$(fixture_record "$name")"
    record="${record:-$base}"
    if [[ "$record" == "$host_toolchain" ]]; then
      compile_fixture "$name"
      for file in "$name.ll" "$name.air.ll"; do
        if ! cmp -s "$out/$file" "$dir/$file"; then
          echo "stale: $file" >&2
          status=1
        fi
      done
      checked=$((checked + 1))
    else
      echo "not checked: $name (compiled by $record)"
      unchecked=$((unchecked + 1))
    fi
    directive_flags "$source"
    # Each committed module must record the Metal version and SDK its TOOLCHAIN record names.
    metal_version="${record#metal: }"
    metal_version="${metal_version%%; macosx-sdk: *}"
    sdk_version="${record##*; macosx-sdk: }"
    sdk_version="${sdk_version%% *}"
    for file in "$name.ll" "$name.air.ll"; do
      if [[ ! -f "$dir/$file" ]]; then
        echo "missing: $file" >&2
        status=1
        continue
      fi
      recorded="$(module_toolchain "$dir/$file")"
      module_metal="${recorded%%|*}"
      module_sdk="${recorded#*|}"
      if [[ "$module_metal" != "$metal_version" ]] || ! same_version "$module_sdk" "$sdk_version"
      then
        echo "toolchain mismatch: $file records ${module_metal:-no Metal version}, SDK" \
          "${module_sdk:-none}; TOOLCHAIN records $record" >&2
        status=1
      fi
      # Each committed module must target the macOS version its source selects, so a compile
      # that took the host's default instead fails here rather than drifting.
      module_target="$(module_macos "$dir/$file")"
      if [[ -z "$module_target" ]] || ! same_version "$module_target" "$macos_min"; then
        echo "deployment target mismatch: $file targets macOS ${module_target:-nothing}; $name.metal" \
          "selects $macos_min ($macos_min_from)" >&2
        status=1
      fi
    done
  done
  if [[ "$(grep -E '^(llvm|llvm-dis|llvm-as): ' "$committed")" != "$(llvm_lines)" ]]; then
    echo "stale: TOOLCHAIN" >&2
    status=1
  fi
  # write_toolchain sorts the fixture lines, so another order is a file written by a run that
  # followed its host's collation instead.
  recorded="$(grep '^fixture ' "$committed" || true)"
  if [[ -n "$recorded" && "$recorded" != "$(printf '%s\n' "$recorded" | sort)" ]]; then
    echo "out of order: TOOLCHAIN fixture lines; rerun this script to rewrite them" >&2
    status=1
  fi
  while IFS= read -r name; do
    [[ -f "$dir/$name.metal" ]] || { echo "orphan: TOOLCHAIN fixture $name" >&2; status=1; }
  done < <(sed -n 's/^fixture \([^:]*\): .*/\1/p' "$committed")
  for file in "$dir"/*.ll; do
    base_name="$(basename "$file")"
    base_name="${base_name%.ll}"
    base_name="${base_name%.air}"
    [[ -f "$dir/$base_name.metal" ]] || { echo "orphan: $(basename "$file")" >&2; status=1; }
  done
  counts="${#sources[@]} sources: $checked checked, $unchecked not checked"
  if [[ $status -eq 0 ]]; then
    echo "air fixtures up to date ($counts)"
  else
    echo "air fixtures not up to date ($counts)" >&2
  fi
  exit $status
fi

if [[ ${#names[@]} -eq 0 ]]; then
  for source in "${sources[@]}"; do
    compile_fixture "$(basename "$source" .metal)"
  done
  : >"$work/fixture-lines"
  write_toolchain "$host_metal" "$host_sdk" "$work/fixture-lines" >"$out/TOOLCHAIN"
  cp "$out"/* "$dir"/
  echo "regenerated ${#sources[@]} fixtures in ${dir#"$repo"/}"
  exit 0
fi

# A partial regeneration keeps the base toolchain and every other fixture's record, and records
# the named fixtures under the host's toolchain.
if [[ ! -f "$committed" ]]; then
  echo "no TOOLCHAIN in ${dir#"$repo"/} to record a partial regeneration in; run without" \
    "names to regenerate every fixture" >&2
  exit 1
fi
if [[ "$(grep -E '^(llvm|llvm-dis|llvm-as): ' "$committed")" != "$(llvm_lines)" ]]; then
  echo "the host's LLVM tools are not the ones TOOLCHAIN records; every fixture shares one" \
    "LLVM record, so run without names to regenerate every fixture" >&2
  exit 1
fi
base_metal="$(toolchain_field metal)"
base_sdk="$(toolchain_field macosx-sdk)"
for name in "${names[@]}"; do
  compile_fixture "$name"
done
: >"$work/fixture-lines"
for source in "${sources[@]}"; do
  name="$(basename "$source" .metal)"
  named=0
  for wanted in "${names[@]}"; do
    if [[ "$wanted" == "$name" ]]; then named=1; fi
  done
  if [[ $named -eq 1 ]]; then
    if [[ "$host_toolchain" != "metal: $base_metal; macosx-sdk: $base_sdk" ]]; then
      echo "fixture $name: $host_toolchain" >>"$work/fixture-lines"
    fi
  else
    record="$(fixture_record "$name")"
    if [[ -n "$record" ]]; then echo "fixture $name: $record" >>"$work/fixture-lines"; fi
  fi
done
write_toolchain "$base_metal" "$base_sdk" "$work/fixture-lines" >"$out/TOOLCHAIN"
cp "$out"/* "$dir"/
echo "regenerated ${#names[@]} of ${#sources[@]} fixtures in ${dir#"$repo"/} ($host_toolchain)"
