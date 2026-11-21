#!/bin/bash
# Probe (never landed): build the release command line and libprobe, then list, size and
# translate every kernel of DATA/lib.metallib under each named committed profile.
# usage: probe/run.sh DATA STEM...
set -u
DATA=$1
shift
WS=$PWD
echo "node $POTLUCK_NODE ws $WS sha $(git rev-parse --short HEAD)"
echo "tools: $(command -v llvm-dis) / $(command -v spirv-val) / $(llvm-dis --version 2>&1 | grep -i version | head -1)"
export CARGO_BUILD_JOBS=8
export METAL2VULKAN_REPRO_DIR="$DATA/repros"
nice -n 10 cargo build --release -p metal2vulkan --features cli --bin metal2vulkan --example libprobe 2>&1 | tail -3
[ -x target/release/metal2vulkan ] || { echo "build failed"; exit 1; }
OUT="$DATA/out"
rm -rf "$OUT"
mkdir -p "$OUT"
PROFILES=()
for s in "$@"; do PROFILES+=("crates/metal2vulkan-validation/profiles/$s.json"); done
target/release/examples/libprobe "$DATA/lib.metallib" "$OUT" "${PROFILES[@]}" || exit 1
while IFS=$'\t' read -r i h name; do
    bc=$(printf '%s/k%03d.bc' "$OUT" "$i")
    if llvm-dis "$bc" -o "$OUT/k.ll" 2>"$OUT/dis.err"; then
        r=$(python3 -I probe/allocas.py "$OUT/k.ll" 2>&1 | tail -1)
    else
        r="dis-failed: $(head -c 200 "$OUT/dis.err")"
    fi
    echo "allocas k$i $h $r"
done < "$OUT/kernels.tsv"
python3 -I probe/xlate.py "$WS/target/release/metal2vulkan" "$DATA/lib.metallib" "$OUT" "$@"
rm -f "$OUT"/k*.bc "$OUT/k.ll" "$OUT/k.spv"
echo done
