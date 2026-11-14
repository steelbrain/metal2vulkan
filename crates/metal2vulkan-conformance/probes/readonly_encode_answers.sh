#!/bin/sh
# Build readonly_encode_answers.m and run every shape over both backings, one shape a process,
# RUNS runs each (default 2), printing each process's report, exit status and the tail of its
# standard error (Metal's assertion text where a call ended the process).
set -u
here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d "${TMPDIR:-/tmp}/ro-encode.XXXXXX")
trap 'rm -f "$work/probe" "$work/err"; rmdir "$work"' EXIT
clang -fobjc-arc -framework Metal -framework Foundation -o "$work/probe" \
  "$here/readonly_encode_answers.m" || exit 1
runs=${RUNS:-2}
for shape in ${SHAPES:-store resolve-into resolve-late custom-depth custom-depth-late \
  kernel-write blit-buffer-into blit-texture-into replace store-then-read}; do
  for backing in bit24 plain; do
    i=0
    while [ "$i" -lt "$runs" ]; do
      echo "=== $shape $backing run $i"
      "$work/probe" "$backing" "$shape" 2>"$work/err"
      echo "exit=$?"
      tail -n 4 "$work/err" | sed 's/^/stderr: /'
      i=$((i + 1))
    done
  done
done
