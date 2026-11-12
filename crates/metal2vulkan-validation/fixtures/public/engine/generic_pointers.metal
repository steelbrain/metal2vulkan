// metal-std: metal4.1
// metal-macos-min: 27.0
// Owned fixture: MSL 4.1's generic address space. A pointer declared with no address space is
// generic (AIR's `addrspace(8)`): a device, thread or threadgroup pointer converts to it, and
// `is_device_addrspace`, `is_thread_addrspace` and `is_threadgroup_addrspace`
// (`air.is_{device,thread,threadgroup}_pointer`) tell which space it points into. The functions
// `space`, `weigh` and `bump` take generic pointers and stay calls (`noinline`).
//
// `space(p)` is 1 for a device pointer, 2 for a thread pointer, 3 for a threadgroup pointer and 4
// for none of them; `weigh(p, k) = p[k] + 100 * space(p)`; `bump(p, k, x)` adds x to p[k].
//
// `generic_spaces`: eight threads in one threadgroup over `in[i] = i` (16 words). Thread t writes
// out[8t + j] for j < 8, each through a generic pointer: [0] weigh(in, t) = t + 100; [1]
// weigh(local, t & 3) after bump(local, t & 3, 0.5), with local[i] = t + i: t + (t & 3) + 200.5;
// [2] weigh(shared, (t + 1) & 7) after a barrier, with shared[l] = 2 * in[l]: 2 * ((t + 1) & 7) +
// 300; [3] weigh(p, t) with p = in + 8 for odd t and in for even t, two device pointers a select
// joins: t + 8 * (t & 1) + 100; [4] 2, then bump(out + 8t, 4, 1.5): 3.5; [5] space(shared) = 3;
// [6] space(local) = 2; [7] space(in) = 1.
//
// `generic_mixed`: thread t of eight picks p = local (local = {10, 11}) where in[t] < 4 and p = in
// otherwise, a generic pointer into the thread or the device space as the run decides, and writes
// out[t] = weigh(p, t & 1): 210 + (t & 1) for t < 4, (t & 1) + 100 for the rest.
//
// `generic_null`: thread t of eight keeps its pointers where in[t] < 4 and makes them null
// otherwise, and writes out[4t + j]: [0] space(in or null), [1] space(local or null), [2]
// space(shared or null), [3] space(nullptr): 1, 2, 3, 4 for t < 4, and 4 four times for the rest
// (a null generic pointer is in no space).
//
// The locals are declared ahead of the switch's only label (on t >> 31, zero for every thread),
// so every path into the switch bypasses their declarations and Clang emits no lifetime markers
// for them (as in ob2_sized_array.metal). Loop-free.
#include <metal_stdlib>
using namespace metal;

__attribute__((noinline)) float space(const float *p) {
    return is_device_addrspace(p)        ? 1.0f
           : is_thread_addrspace(p)      ? 2.0f
           : is_threadgroup_addrspace(p) ? 3.0f
                                         : 4.0f;
}

__attribute__((noinline)) float weigh(const float *p, uint k) {
    return p[k] + 100.0f * space(p);
}

__attribute__((noinline)) void bump(float *p, uint k, float x) {
    p[k] += x;
}

kernel void generic_spaces(device float *out [[buffer(0)]],
                           device const float *in [[buffer(1)]],
                           uint t [[thread_position_in_grid]],
                           uint l [[thread_position_in_threadgroup]]) {
    threadgroup float shared[8];
    switch (t >> 31) {
        float local[4];
    default:
        shared[l] = 2.0f * in[l];
        local[0] = float(t);
        local[1] = float(t) + 1.0f;
        local[2] = float(t) + 2.0f;
        local[3] = float(t) + 3.0f;
        bump(local, t & 3, 0.5f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        device const float *p = (t & 1) ? in + 8 : in;
        device float *o = out + 8 * t;
        o[0] = weigh(in, t);
        o[1] = weigh(local, t & 3);
        o[2] = weigh(shared, (t + 1) & 7);
        o[3] = weigh(p, t);
        o[4] = 2.0f;
        bump(o, 4, 1.5f);
        o[5] = space(shared);
        o[6] = space(local);
        o[7] = space(in);
    }
}

kernel void generic_mixed(device float *out [[buffer(0)]],
                          device const float *in [[buffer(1)]],
                          uint t [[thread_position_in_grid]]) {
    switch (t >> 31) {
        float local[2];
    default:
        local[0] = 10.0f;
        local[1] = 11.0f;
        const float *p = in[t] < 4.0f ? (const float *)local : (const float *)in;
        out[t] = weigh(p, t & 1);
    }
}

kernel void generic_null(device float *out [[buffer(0)]],
                         device const float *in [[buffer(1)]],
                         uint t [[thread_position_in_grid]],
                         uint l [[thread_position_in_threadgroup]]) {
    threadgroup float shared[8];
    switch (t >> 31) {
        float local[2];
    default:
        local[0] = 1.0f;
        shared[l] = 1.0f;
        bool keep = in[t] < 4.0f;
        out[4 * t + 0] = space(keep ? in : nullptr);
        out[4 * t + 1] = space(keep ? (const float *)local : nullptr);
        out[4 * t + 2] = space(keep ? (const float *)shared : nullptr);
        out[4 * t + 3] = space(nullptr);
    }
}
