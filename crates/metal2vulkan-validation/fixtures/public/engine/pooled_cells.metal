// Owned fixture: thread pointers held in a thread's array of structs at indices only the run knows,
// among the structs' data. `closure_alloc` hands out the next element of the array, or null once
// its room is used up; `closure_alloc_extra` takes room from the array's end and hands out a
// thread pointer to it, which the element keeps in its `extra` field. Each kernel runs eight
// threads in one threadgroup over `in[i] = i + 0.5`.
//
// `pooled_closures`: thread t's array has room for 2 + (t & 3) elements. It asks three times for
// an element (type j + 1, weight in[t] + j), each followed by extra room (10 (j + 1), in[t]) where
// some is left, and writes out[8t + j]: [0..3] element j's weight plus its extra's first float, its
// weight alone without extra room, -1 past the elements handed out; [3] the elements handed out;
// [4] the room left; [5] the sum of their types; [6] the sum of their extras' second floats; [7]
// the elements with extra room.
//
// `pooled_bytes` and `pooled_spill`: a thread pointer to vals[t & 3] (vals[k] = in[t] + k, another
// array) is copied into an element's `extra` as data: as bytes, a byte loop over a length the run
// reads (`len[0]` = 8) from a struct that holds the pointer; and as an integer of its bits, stored
// into device memory (`spill[t]`, volatile) and read back. Words [0] the float the copied pointer
// reads; [1] vals[t & 3] after a write of +100 through it; [2] 1 where the copied pointer equals
// &vals[t & 3]; [3] the elements handed out; the rest 0.
//
// `pooled_context`: the closures of `pooled_closures` behind a context struct that holds a
// constant pointer to an argument buffer (`Anc`, whose `held` points at device structs of
// textures), as a renderer's kernels keep theirs; `shade` (noinline) reads texture `held[t & 1]`
// at x = t & 3 through the context. Words [0] the texel's red, [1..7] as `pooled_closures`' [1..7].
//
// Each array is declared ahead of its switch's only label (on t >> 31, zero for every thread), so
// every path into the switch bypasses its declaration and Clang emits no lifetime markers for it
// (as in generic_pointers.metal).
#include <metal_stdlib>
using namespace metal;

struct Closure {
    float w;
    int type;
    thread float *extra;
    float data[2];
};

struct Shader {
    Closure closure[6];
    int num;
    int left;
};

static thread Closure *closure_alloc(thread Shader &sd, int type, float w) {
    if (sd.left == 0)
        return nullptr;
    thread Closure *sc = &sd.closure[sd.num];
    sc->type = type;
    sc->w = w;
    sc->extra = nullptr;
    sd.num++;
    sd.left--;
    return sc;
}

static thread float *closure_alloc_extra(thread Shader &sd) {
    if (sd.left == 0)
        return nullptr;
    sd.left--;
    return &sd.closure[sd.num + sd.left].data[0];
}

// Three elements, each with extra room where some is left.
static void fill(thread Shader &sd, float x) {
    for (int j = 0; j < 3; j++) {
        thread Closure *c = closure_alloc(sd, j + 1, x + float(j));
        if (c == nullptr)
            continue;
        thread float *e = closure_alloc_extra(sd);
        if (e != nullptr) {
            e[0] = float(10 * (j + 1));
            e[1] = x;
            c->extra = e;
        }
    }
}

// Words [first..8) of `pooled_closures`.
static void summarize(thread Shader &sd, device float *o, int first) {
    int types = 0, extras = 0;
    float seconds = 0.0f;
    for (int i = 0; i < 3; i++) {
        float v = -1.0f;
        if (i < sd.num) {
            thread float *e = sd.closure[i].extra;
            v = sd.closure[i].w + (e != nullptr ? e[0] : 0.0f);
            types += sd.closure[i].type;
            if (e != nullptr) {
                seconds += e[1];
                extras++;
            }
        }
        if (i >= first)
            o[i] = v;
    }
    o[3] = float(sd.num);
    o[4] = float(sd.left);
    o[5] = float(types);
    o[6] = seconds;
    o[7] = float(extras);
}

kernel void pooled_closures(device float *out [[buffer(0)]],
                            device const float *in [[buffer(1)]],
                            uint t [[thread_position_in_grid]]) {
    switch (t >> 31) {
        Shader sd;
    default:
        sd.num = 0;
        sd.left = 2 + int(t & 3);
        fill(sd, in[t]);
        summarize(sd, out + 8 * t, 0);
    }
}

struct Holder {
    thread float *p;
    int n;
};

kernel void pooled_bytes(device float *out [[buffer(0)]],
                         device const float *in [[buffer(1)]],
                         device const int *len [[buffer(2)]],
                         uint t [[thread_position_in_grid]]) {
    switch (t >> 31) {
        Shader sd;
        float vals[4];
        Holder h;
    default:
        for (int k = 0; k < 4; k++)
            vals[k] = in[t] + float(k);
        h.p = &vals[t & 3];
        h.n = len[0];
        sd.num = int(t & 1);
        sd.left = 2;
        thread Closure *c = closure_alloc(sd, 1, 1.0f);
        thread uchar *d = (thread uchar *)&c->extra;
        thread const uchar *s = (thread const uchar *)&h.p;
        for (int k = 0; k < h.n; k++)
            d[k] = s[k];
        device float *o = out + 8 * t;
        thread float *e = sd.closure[sd.num - 1].extra;
        o[0] = *e;
        *e += 100.0f;
        o[1] = vals[t & 3];
        o[2] = e == &vals[t & 3] ? 1.0f : 0.0f;
        o[3] = float(sd.num);
    }
}

kernel void pooled_spill(device float *out [[buffer(0)]],
                         device const float *in [[buffer(1)]],
                         volatile device ulong *spill [[buffer(2)]],
                         uint t [[thread_position_in_grid]]) {
    switch (t >> 31) {
        Shader sd;
        float vals[4];
    default:
        for (int k = 0; k < 4; k++)
            vals[k] = in[t] + float(k);
        thread float *p = &vals[t & 3];
        ulong bits;
        __builtin_memcpy(&bits, &p, sizeof(p));
        spill[t] = bits;
        sd.num = int(t & 1);
        sd.left = 2;
        thread Closure *c = closure_alloc(sd, 1, 1.0f);
        ulong back = spill[t];
        __builtin_memcpy(&c->extra, &back, sizeof(back));
        device float *o = out + 8 * t;
        thread float *e = sd.closure[sd.num - 1].extra;
        o[0] = *e;
        *e += 100.0f;
        o[1] = vals[t & 3];
        o[2] = e == &vals[t & 3] ? 1.0f : 0.0f;
        o[3] = float(sd.num);
    }
}

struct Held {
    texture2d<float> t;
};

struct Anc {
    device const Held *held;
};

struct Context {
    constant Anc *anc;
    int k;
};

__attribute__((noinline)) float shade(thread Context &ctx, thread Shader &sd, float x, uint t) {
    fill(sd, x);
    return ctx.anc->held[ctx.k].t.read(uint2(t & 3, 0)).x;
}

kernel void pooled_context(device float *out [[buffer(0)]],
                           device const float *in [[buffer(1)]],
                           constant Anc *anc [[buffer(2)]],
                           uint t [[thread_position_in_grid]]) {
    switch (t >> 31) {
        Shader sd;
        Context ctx;
    default:
        ctx.anc = anc;
        ctx.k = int(t & 1);
        sd.num = 0;
        sd.left = 2 + int(t & 3);
        device float *o = out + 8 * t;
        o[0] = shade(ctx, sd, in[t], t);
        summarize(sd, o, 1);
    }
}
