#!/bin/sh
# Run opencl_operand_encodings on this Mac: build the probe, run it in out/, carve each binary it
# writes into its bitcode modules, disassemble them (llvm-dis, else Apple's metal-opt -S), and
# print the lines that name the operands the probe asks about. The .ll files stay in out/.
set -u
P=$(cd "$(dirname "$0")" && pwd)
mkdir -p out && cd out || exit 1
xcrun clang -fobjc-arc -framework Metal -framework Foundation -framework OpenCL \
    -o probe "$P/opencl_operand_encodings.m" || exit 1
./probe
for f in *.bin; do
    [ -e "$f" ] && python3 -I "$P/opencl_operand_encodings_carve.py" "$f"
done
for b in *.bc; do
    [ -e "$b" ] || continue
    l="${b%.bc}.ll"
    llvm-dis "$b" -o "$l" 2>"$l.err" || xcrun metal-opt -S "$b" -o "$l" 2>>"$l.err" ||
        { echo "no disassembly: $b"; head -c 600 "$l.err"; echo; }
done
for l in *.ll; do
    [ -e "$l" ] || continue
    echo "== $l"
    grep -n -E 'air\.(wg\.barrier|simdgroup\.barrier|atomic|sample_texture|read_texture|write_texture|get_read_sampler|sampler)|bitreverse|sampler|triple|air\.version|air\.language_version' "$l" |
        cut -c1-400
done
