// Owned fixture: `%` on signed integers, whose remainder takes the dividend's sign, at every width
// and with operands of both signs, the most negative values among them. Each thread reads two
// 64-bit operands as four words (the dividend's low and high words, then the divisor's) and writes
// eight. A remainder MSL leaves undefined (by zero, or of the most negative value by -1) is never
// taken: its word is 0xdeadbeef instead. Written for this repository; no third-party source.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

// Words 8i to 8i + 7: the low words' remainder as `int`, as `short` and as `char` (each
// zero-extended); the 64-bit remainder's low and high words; the exclusive-or of the lanes of the
// `int2` remainder of (low, high) by (low, high); the low word's remainders by the constants 3 and
// -8.
kernel void signed_remainders(device const uint *a [[buffer(0)]],
                              device uint *o [[buffer(1)]],
                              uint i [[thread_position_in_grid]]) {
    const uint marker = 0xdeadbeefu;
    const int int_min = int(0x80000000u);
    const long long_min = long(0x8000000000000000ul);
    int xl = int(a[4 * i + 0]);
    int xh = int(a[4 * i + 1]);
    int yl = int(a[4 * i + 2]);
    int yh = int(a[4 * i + 3]);
    long x = long((ulong(uint(xh)) << 32) | ulong(uint(xl)));
    long y = long((ulong(uint(yh)) << 32) | ulong(uint(yl)));
    bool low_ok = yl != 0 && !(xl == int_min && yl == -1);
    bool high_ok = yh != 0 && !(xh == int_min && yh == -1);
    bool long_ok = y != 0 && !(x == long_min && y == -1);
    short xs = short(xl);
    short ys = short(yl);
    char xc = char(xl);
    char yc = char(yl);
    o[8 * i + 0] = low_ok ? uint(xl % yl) : marker;
    o[8 * i + 1] = ys != 0 ? uint(ushort(short(xs % ys))) : marker;
    o[8 * i + 2] = yc != 0 ? uint(uchar(char(xc % yc))) : marker;
    ulong r = long_ok ? ulong(x % y) : ulong(marker);
    o[8 * i + 3] = uint(r);
    o[8 * i + 4] = uint(r >> 32);
    int2 v = low_ok && high_ok ? int2(xl, xh) % int2(yl, yh) : int2(int(marker), 0);
    o[8 * i + 5] = uint(v.x ^ v.y);
    o[8 * i + 6] = uint(xl % 3);
    o[8 * i + 7] = uint(xl % -8);
}
