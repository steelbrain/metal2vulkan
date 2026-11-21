# Probe (never landed): translate every kernel of a library under each device file, one release
# command-line run each, with its wall time and peak RSS; prints lines with no function name.
# Kernels refused for IdentityDomains run again with the switch form forced (M2V_PROBE_SWITCH) and
# under the committed profile's resident tables (--profile).
# usage: python3 -I xlate.py CLI LIB OUT STEM...
import os
import re
import subprocess
import sys
import time
from collections import Counter

cli, lib, out = sys.argv[1:4]
stems = sys.argv[4:]
kernels = [l.rstrip("\n").split("\t") for l in open(os.path.join(out, "kernels.tsv"))]
logs = os.path.join(out, "logs")
os.makedirs(logs, exist_ok=True)


def run(stem, mode, index, short, name, args, env):
    log = os.path.join(logs, f"{stem}.{mode}.k{index}.txt")
    with open(log, "w") as f:
        start = time.monotonic()
        p = subprocess.Popen(
            [cli, lib, os.path.join(out, "k.spv"), *args, "--function", name],
            stdout=f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
        )
        _, status, usage = os.wait4(p.pid, 0)
        wall = time.monotonic() - start
    code = os.waitstatus_to_exitcode(status)
    lines = [l.rstrip("\n").replace(name, f"<k{index}>") for l in open(log)]
    cands = Counter(l.split()[1] for l in lines if l.startswith("probe-heap-array-candidates"))
    weights = [l for l in lines if l.startswith("probe-weight") or l.startswith("probe-past")]
    keep = [l for l in lines if l.startswith("metal2vulkan:") or "PASS" in l]
    keep = [re.sub(r"wrote \S+", "wrote", l) for l in keep]
    verdict = " | ".join(keep)[-600:]
    c = ",".join(f"{k}x{v}" for k, v in sorted(cands.items()))
    print(
        f"xlate {stem} {mode} k{index} {short} rc={code} wall={wall:.2f}s "
        f"peak={usage.ru_maxrss / 1024:.0f}MiB weights=[{'; '.join(weights)}] cands=[{c}] :: {verdict}",
        flush=True,
    )
    os.remove(log)
    return "IdentityDomains" in verdict


base = dict(os.environ)
switch = dict(os.environ, M2V_PROBE_SWITCH="1")
for stem in stems:
    device = os.path.join(out, stem + ".device.json")
    profile = os.path.join("crates/metal2vulkan-validation/profiles", stem + ".json")
    identity = []
    for index, short, name in kernels:
        if run(stem, "identity", index, short, name, ["--device", device], base):
            identity.append((index, short, name))
    for index, short, name in identity:
        run(stem, "switch", index, short, name, ["--device", device], switch)
        run(stem, "resident", index, short, name, ["--profile", profile], base)
