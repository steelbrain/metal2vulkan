// Owned fixture: a thread pointer that may be null. `closure_alloc` hands out the next element of a
// thread's array at an index only the run knows, or null once the array's room is used up, so its
// result is a `phi` of an element's address and null; `weigh` (`noinline`, so the null check stays
// on the pointer) asks whether it is null, and writes through it where it is not.
//
// Eight threads over `in[i] = i + 0.5`. Thread t's array has room for t & 3 closures. It asks for
// four, then a fifth, and writes out[8t + j]: [0..4] weigh(closure_alloc(sd, j + 1), in[t] + j) =
// 2 * (in[t] + j) for j < (t & 3) and -1 for the rest (null); [4] the closures handed out, t & 3;
// [5] the sum of their types, 1 + 2 + ... + (t & 3); [6] the sum of their data, in[t] + j each;
// [7] 1 where the fifth request is null (every thread: at most three fit).
//
// The array is declared ahead of the switch's only label (on t >> 31, zero for every thread), so
// every path into the switch bypasses its declaration and Clang emits no lifetime markers for it
// (as in generic_pointers.metal).
#include <metal_stdlib>
using namespace metal;

struct Closure {
    float w;
    int type;
    float data[2];
};

struct Shader {
    Closure closure[4];
    int num;
    int left;
};

static thread Closure *closure_alloc(thread Shader &sd, int type) {
    if (sd.left == 0)
        return nullptr;
    thread Closure *sc = &sd.closure[sd.num];
    sc->type = type;
    sd.num++;
    sd.left--;
    return sc;
}

__attribute__((noinline)) float weigh(thread Closure *c, float x) {
    if (c == nullptr)
        return -1.0f;
    c->data[0] = x;
    c->w = 2.0f * x;
    return c->w;
}

__attribute__((noinline)) float is_null(thread Closure *c) {
    return c == nullptr ? 1.0f : 0.0f;
}

kernel void nullable_private(device float *out [[buffer(0)]],
                             device const float *in [[buffer(1)]],
                             uint t [[thread_position_in_grid]]) {
    switch (t >> 31) {
        Shader sd;
    default:
        sd.num = 0;
        sd.left = int(t & 3);
        device float *o = out + 8 * t;
        for (int j = 0; j < 4; j++)
            o[j] = weigh(closure_alloc(sd, j + 1), in[t] + float(j));
        o[4] = float(sd.num);
        int types = 0;
        float data = 0.0f;
        for (int i = 0; i < sd.num; i++) {
            types += sd.closure[i].type;
            data += sd.closure[i].data[0];
        }
        o[5] = float(types);
        o[6] = data;
        o[7] = is_null(closure_alloc(sd, 9));
    }
}
