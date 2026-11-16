#!/bin/sh
# Runs simd_matrix16.m on this Mac's Metal over the committed fixture
# engine/simd_matrix16.air.ll: assembles its typed text into a metallib, builds the probe, and runs
# it once per form named, one process each, so that a compiler failure on one form leaves the
# others. Scratch files live in a temporary directory removed on exit.
#
# Usage: sh crates/metal2vulkan-conformance/probes/simd_matrix16.sh <form>...
#   (forms: f32_f32 f16_f16 s_s)
set -u
unset MTL_DEBUG_LAYER
here=$(cd "$(dirname "$0")" && pwd)
fixture="$here/../../metal2vulkan-validation/fixtures/public/engine/simd_matrix16.air.ll"
work=$(mktemp -d "${TMPDIR:-/tmp}/simd_matrix16.XXXXXX") || exit 2
trap 'rm -rf "$work"' EXIT
sw_vers
xcrun -sdk macosx metal --version | head -1
xcrun -sdk macosx metal-as "$fixture" -o "$work/m.air" || exit 2
xcrun -sdk macosx metallib "$work/m.air" -o "$work/m.metallib" || exit 2
clang -fobjc-arc -O1 -fmodules -o "$work/probe" "$here/simd_matrix16.m" \
  -framework Foundation -framework Metal || exit 2
status=0
for form in "$@"; do
  "$work/probe" "$work/m.metallib" "$form"
  code=$?
  echo "EXIT $form=$code"
  [ "$code" -eq 0 ] || status=1
done
exit $status
