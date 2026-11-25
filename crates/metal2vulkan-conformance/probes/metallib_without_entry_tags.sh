#!/bin/sh
# Runs metallib_without_entry_tags.m on this Mac's Metal: builds the probe, then for each case
# named compiles the MSL below into a metallib with Apple's toolchain (`metal -c`, `metallib`) and
# runs the case once with MTL_DEBUG_LAYER=1 and once without, one process each. Every run compiles
# with its own SEED, which every function's result carries, so no run is answered from functions
# another run compiled. Scratch files live in a temporary directory removed on exit.
#
# Usage: sh crates/metal2vulkan-conformance/probes/metallib_without_entry_tags.sh <case>...
#   (cases: as-is, v2.2, no-TAG, no-TAG-shift, no-TAG-v2.2; TAG one of NAME TYPE HASH MDSZ OFFT VERS)
set -u
unset MTL_DEBUG_LAYER
here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d "${TMPDIR:-/tmp}/metallib_without_entry_tags.XXXXXX") || exit 2
trap 'rm -rf "$work"' EXIT
sw_vers
xcrun -sdk macosx metal --version | head -1
clang -fobjc-arc -O1 -fmodules -o "$work/probe" "$here/metallib_without_entry_tags.m" \
  -framework Foundation -framework Metal || exit 2
cat > "$work/k.metal" <<'EOF'
#include <metal_stdlib>
using namespace metal;
kernel void fill_a(device uint *o [[buffer(0)]], uint t [[thread_position_in_grid]]) {
  o[t] = t * 3 + SEED;
}
kernel void fill_b(device uint *o [[buffer(0)]], uint t [[thread_position_in_grid]]) {
  o[t] = t * 5 + SEED;
}
struct Out {
  float4 p [[position]];
  float c;
};
vertex Out vert(uint v [[vertex_id]]) {
  float2 q = float2((v << 1) & 2, v & 2);
  Out o;
  o.p = float4(q * 2.0 - 1.0, 0.0, 1.0);
  o.c = float(SEED % 251u) / 250.0;
  return o;
}
fragment float4 frag(Out in [[stage_in]]) {
  return float4(in.c, float(SEED % 13u) / 12.0, 0.25, 1.0);
}
EOF
base=$(( $(date +%s) % 1000000 * 100 ))
n=0
status=0
for c in "$@"; do
  for layer in on off; do
    n=$((n + 1))
    seed=$((base + n))
    xcrun -sdk macosx metal -DSEED="${seed}u" -c "$work/k.metal" -o "$work/k.air" || exit 2
    xcrun -sdk macosx metallib "$work/k.air" -o "$work/k.metallib" || exit 2
    if [ "$layer" = on ]; then
      MTL_DEBUG_LAYER=1 "$work/probe" "$work/k.metallib" "$seed" "$c"
    else
      "$work/probe" "$work/k.metallib" "$seed" "$c"
    fi
    code=$?
    echo "EXIT $c layer=$layer code=$code"
    [ "$code" -eq 0 ] || status=1
  done
done
exit $status
