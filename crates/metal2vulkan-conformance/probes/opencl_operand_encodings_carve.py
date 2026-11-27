"""Carve the LLVM bitcode modules out of a compiled program's bytes (opencl_operand_encodings).

    python3 -I opencl_operand_encodings_carve.py FILE...

Each module found in FILE, a bitcode wrapper (0x0B17C0DE, its offset and size from its header) or
a raw stream ('BC' 0xC0DE, its end where its top-level blocks end), is written to
FILE's stem + '.<n>.bc', and one line per module is printed.
"""

import struct
import sys

WRAPPER = b"\xde\xc0\x17\x0b"
RAW = b"BC\xc0\xde"
# BLOCKINFO, MODULE, IDENTIFICATION, STRTAB, SYMTAB.
TOP_BLOCKS = {0, 8, 13, 23, 25}


class Bits:
    def __init__(self, data, start):
        self.data = data
        self.pos = start * 8

    def read(self, n):
        v = 0
        for k in range(n):
            i = self.pos + k
            if i // 8 >= len(self.data):
                raise EOFError
            v |= ((self.data[i // 8] >> (i % 8)) & 1) << k
        self.pos += n
        return v

    def vbr(self, n):
        v, shift = 0, 0
        while True:
            chunk = self.read(n)
            v |= (chunk & ((1 << (n - 1)) - 1)) << shift
            shift += n - 1
            if not chunk >> (n - 1):
                return v

    def align32(self):
        self.pos = (self.pos + 31) // 32 * 32


def raw_end(data, start):
    """The byte offset where the raw stream at `start` ends."""
    bits = Bits(data, start + 4)
    end = bits.pos
    while True:
        try:
            if bits.read(2) != 1:
                break
            block = bits.vbr(8)
            bits.vbr(4)
            bits.align32()
            words = bits.read(32)
        except EOFError:
            break
        if block not in TOP_BLOCKS or bits.pos // 8 + 4 * words > len(data):
            break
        bits.pos += 32 * words
        end = bits.pos
    return end // 8


def carve(path):
    data = open(path, "rb").read()
    stem = path.rsplit(".", 1)[0]
    found, covered = [], []
    at = data.find(WRAPPER)
    while at >= 0:
        if at + 20 <= len(data):
            _, _, offset, size, _ = struct.unpack_from("<5I", data, at)
            if at + offset + size <= len(data):
                found.append((at, data[at + offset : at + offset + size]))
                covered.append((at, at + offset + size))
        at = data.find(WRAPPER, at + 1)
    at = data.find(RAW)
    while at >= 0:
        if not any(a <= at < b for a, b in covered):
            end = raw_end(data, at)
            found.append((at, data[at:end]))
            covered.append((at, end))
        at = data.find(RAW, at + 1)
    for n, (at, module) in enumerate(found):
        out = f"{stem}.{n}.bc"
        with open(out, "wb") as f:
            f.write(module)
        print(f"carved {path} at={at} bytes={len(module)} -> {out}")
    if not found:
        print(f"carved {path} none (bytes={len(data)})")


for p in sys.argv[1:]:
    carve(p)
