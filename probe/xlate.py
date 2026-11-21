# Probe (never landed): translate every kernel of a library under each device file, one release
# command-line run each, with its wall time and peak RSS; prints lines with no function name.
# usage: python3 -I xlate.py CLI LIB OUT STEM...
import os
import subprocess
import sys
import time

cli, lib, out = sys.argv[1:4]
stems = sys.argv[4:]
kernels = [l.rstrip("\n").split("\t") for l in open(os.path.join(out, "kernels.tsv"))]
logs = os.path.join(out, "logs")
os.makedirs(logs, exist_ok=True)
for stem in stems:
    device = os.path.join(out, stem + ".device.json")
    for index, short, name in kernels:
        log = os.path.join(logs, f"{stem}.k{index}.txt")
        with open(log, "w") as f:
            start = time.monotonic()
            p = subprocess.Popen(
                [cli, lib, os.path.join(out, "k.spv"), "--device", device, "--function", name],
                stdout=f,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
            _, status, usage = os.wait4(p.pid, 0)
            wall = time.monotonic() - start
        code = os.waitstatus_to_exitcode(status)
        lines = [l.rstrip("\n").replace(name, f"<k{index}>") for l in open(log)]
        lines = [l for l in lines if l.startswith("metal2vulkan:") or "PASS" in l]
        verdict = " | ".join(lines)[-900:]
        print(
            f"xlate {stem} k{index} {short} rc={code} wall={wall:.2f}s "
            f"peak={usage.ru_maxrss / 1024:.0f}MiB :: {verdict}",
            flush=True,
        )
