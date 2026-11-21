# Probe (never landed): the allocas of an LLVM text module, their sizes, and the bytes live at
# once by the module's llvm.lifetime.start/end markers, walked in text order per function.
# usage: python3 -I allocas.py FILE.ll
# prints: allocas N sum S peak P top a,b,c,d,e unmarked U callees C
import re
import sys

text = open(sys.argv[1]).read()
structs = {}
for m in re.finditer(r"^(%[\w.$\"-]+) = type (.*)$", text, re.M):
    structs[m.group(1)] = m.group(2).strip()


def pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


class P:
    def __init__(self, s):
        self.s = s
        self.i = 0

    def ws(self):
        while self.i < len(self.s) and self.s[self.i] == " ":
            self.i += 1

    def peek(self, t):
        self.ws()
        return self.s.startswith(t, self.i)

    def take(self, t):
        self.ws()
        assert self.s.startswith(t, self.i), (self.s, self.i, t)
        self.i += len(t)

    def num(self):
        self.ws()
        m = re.compile(r"\d+").match(self.s, self.i)
        self.i = m.end()
        return int(m.group(0))

    def ty(self):
        """(size, align) of the type at the cursor."""
        self.ws()
        s = self.s
        if self.peek("<{"):
            self.take("<{")
            return self.fields("}>", packed=True)
        if self.peek("{"):
            self.take("{")
            return self.fields("}", packed=False)
        if self.peek("<"):
            self.take("<")
            n = self.num()
            self.take("x")
            es, ea = self.ty()
            self.take(">")
            size = n * es
            a = pow2(size)
            return a, a
        if self.peek("["):
            self.take("[")
            n = self.num()
            self.take("x")
            es, ea = self.ty()
            self.take("]")
            return n * es, ea
        m = re.compile(r"(%[\w.$\"-]+|ptr( addrspace\(\d+\))?|i\d+|half|bfloat|float|double)").match(s, self.i)
        assert m, (s, self.i)
        self.i = m.end()
        t = m.group(1)
        # Typed pointers: T*.
        while self.peek("*"):
            self.take("*")
            t = "ptr"
        if t.startswith("ptr"):
            return 8, 8
        if t.startswith("%"):
            return P(structs[t]).ty()
        if t in ("half", "bfloat"):
            return 2, 2
        if t == "float":
            return 4, 4
        if t == "double":
            return 8, 8
        bits = int(t[1:])
        b = pow2(max(1, (bits + 7) // 8))
        return b, b

    def fields(self, close, packed):
        off, align = 0, 1
        if self.peek(close):
            self.take(close)
            return 0, 1
        while True:
            fs, fa = self.ty()
            if packed:
                fa = 1
            off = (off + fa - 1) // fa * fa + fs
            align = max(align, fa)
            if self.peek(","):
                self.take(",")
                continue
            self.take(close)
            break
        return (off + align - 1) // align * align, align


def alloc_size(t):
    s, a = P(t).ty()
    return (s + a - 1) // a * a


allocas = []  # (function, name, size)
peaks = []
unmarked_total = 0
functions = re.split(r"^define ", text, flags=re.M)[1:]
for body in functions:
    end = body.find("\n}")
    body = body[:end]
    sizes = {}
    order = []
    for m in re.finditer(r"^\s*(%[\w.$\"-]+) = alloca (.*?)(?:, i(?:32|64) (\d+))?(?:, align \d+)?(?:, addrspace\(\d+\))?\s*(?:,.*)?$", body, re.M):
        t = m.group(2)
        count = int(m.group(3) or 1)
        size = alloc_size(t) * count
        sizes[m.group(1)] = size
        order.append(m.group(1))
    marked = set()
    events = []
    for m in re.finditer(r"@llvm\.lifetime\.(start|end)[\w.]*\((?:i64 (-?\d+), )?ptr(?: nonnull)? (%[\w.$\"-]+)\)", body):
        kind, size, name = m.group(1), m.group(2), m.group(3)
        if name in sizes:
            marked.add(name)
            events.append((m.start(), kind, name))
    unmarked = sum(sizes[n] for n in sizes if n not in marked)
    unmarked_total += unmarked
    live, cur, peak = set(), 0, 0
    for _, kind, name in sorted(events):
        if kind == "start" and name not in live:
            live.add(name)
            cur += sizes[name]
        elif kind == "end" and name in live:
            live.discard(name)
            cur -= sizes[name]
        peak = max(peak, cur)
    peaks.append(unmarked + peak)
    allocas.extend(sizes[n] for n in order)

top = sorted(allocas, reverse=True)[:6]
print(
    f"allocas {len(allocas)} sum {sum(allocas)} peak {max(peaks) if peaks else 0} "
    f"top {','.join(map(str, top))} unmarked {unmarked_total} functions {len(functions)}"
)
