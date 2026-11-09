// Owned fixture: MSL's integer `median3`, which Metal's compiler spells
// `air.median3.<s|u>.<type>`, on char, uchar, short, ushort, int, uint, long and ulong, and on
// char4, ushort2, int4, uint4, long2 and ulong2. Each thread t reads three 64-bit words
// a = x[3t], b = x[3t + 1] and c = x[3t + 2] and writes the 32 words 32t .. 32t + 31: each
// scalar type's median of the three words cast to it (a narrow signed one sign-extended to 32
// bits, a 64-bit one as its low then its high word), then each vector's lanes: char4's are the
// medians of the words' four low bytes, ushort2's of their two low halves; int4's and uint4's of
// the words' low and high halves, then of the complements' (~a, ~b, ~c); long2's and ulong2's of
// the words, then of their complements. Written for this repository; no third-party source.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void median3_ops(device const ulong *x [[buffer(0)]],
                        device uint *o [[buffer(1)]],
                        uint t [[thread_position_in_grid]]) {
    ulong a = x[3 * t], b = x[3 * t + 1], c = x[3 * t + 2];
    device uint *r = o + t * 32;
    r[0] = uint(int(median3(char(a), char(b), char(c))));
    r[1] = uint(median3(uchar(a), uchar(b), uchar(c)));
    r[2] = uint(int(median3(short(a), short(b), short(c))));
    r[3] = uint(median3(ushort(a), ushort(b), ushort(c)));
    r[4] = uint(median3(int(a), int(b), int(c)));
    r[5] = median3(uint(a), uint(b), uint(c));
    ulong sl = ulong(median3(long(a), long(b), long(c)));
    r[6] = uint(sl);
    r[7] = uint(sl >> 32);
    ulong ul = median3(a, b, c);
    r[8] = uint(ul);
    r[9] = uint(ul >> 32);
    char4 c4 = median3(as_type<char4>(uint(a)), as_type<char4>(uint(b)), as_type<char4>(uint(c)));
    ushort2 u2 = median3(as_type<ushort2>(uint(a)), as_type<ushort2>(uint(b)),
                         as_type<ushort2>(uint(c)));
    int4 i4 = median3(as_type<int4>(ulong2(a, ~a)), as_type<int4>(ulong2(b, ~b)),
                      as_type<int4>(ulong2(c, ~c)));
    uint4 v4 = median3(as_type<uint4>(ulong2(a, ~a)), as_type<uint4>(ulong2(b, ~b)),
                       as_type<uint4>(ulong2(c, ~c)));
    long2 l2 = median3(long2(long(a), long(~a)), long2(long(b), long(~b)),
                       long2(long(c), long(~c)));
    ulong2 w2 = median3(ulong2(a, ~a), ulong2(b, ~b), ulong2(c, ~c));
    r[10] = uint(int(c4.x));
    r[11] = uint(int(c4.y));
    r[12] = uint(int(c4.z));
    r[13] = uint(int(c4.w));
    r[14] = u2.x;
    r[15] = u2.y;
    r[16] = uint(i4.x);
    r[17] = uint(i4.y);
    r[18] = uint(i4.z);
    r[19] = uint(i4.w);
    r[20] = v4.x;
    r[21] = v4.y;
    r[22] = v4.z;
    r[23] = v4.w;
    r[24] = uint(ulong(l2.x));
    r[25] = uint(ulong(l2.x) >> 32);
    r[26] = uint(ulong(l2.y));
    r[27] = uint(ulong(l2.y) >> 32);
    r[28] = uint(w2.x);
    r[29] = uint(w2.x >> 32);
    r[30] = uint(w2.y);
    r[31] = uint(w2.y >> 32);
}
