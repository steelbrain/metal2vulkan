// Owned fixture: a thread's array of device pointers, each element an entry buffer argument, and
// accesses through an element at an index only the run knows (`pick`, `count`, a constant
// buffer's words). Buffer 3 is the output in each kernel.
// - pointer_array_pick: out[i] = srcs[(pick + i) % 3][i] over srcs = {a, b, c}; nothing else
//   reaches the array.
// - pointer_array_by_value: the array's address travels in an `array_ref` passed by value to a
//   `noinline` function, which reads word i through element (pick + i) % 2 of {a, b, out} and
//   writes it, plus i, through element `to` (2, the output, in the cases).
// - pointer_array_zero_filled: a zeroed array of 16 pointers holds a, b and c in its first `count`
//   elements (at most 3); each of those elements moves on by i words, and out[i] folds the words
//   the moved elements name, element by element.
#include <metal_stdlib>
using namespace metal;

kernel void pointer_array_pick(device const uint *a [[buffer(0)]],
                               device const uint *b [[buffer(1)]],
                               device const uint *c [[buffer(2)]],
                               device uint *out [[buffer(3)]],
                               constant uint &pick [[buffer(4)]],
                               uint i [[thread_position_in_grid]]) {
    device const uint *srcs[3] = {a, b, c};
    out[i] = srcs[(pick + i) % 3][i];
}

struct Destinations {
    array_ref<device uint *> buffers;
    uint bias;
};

// An external function keeps its signature, and Apple's compiler passes a struct member by
// member, an `array_ref` member as one aggregate value: the kernel inserts the array's address
// into it, and the function extracts it.
__attribute__((noinline)) void move_word(Destinations d, uint from, uint to, uint i) {
    uint n = d.buffers.size();
    d.buffers[to % n][i] = d.buffers[from % n][i] + d.bias;
}

kernel void pointer_array_by_value(device uint *a [[buffer(0)]],
                                   device uint *b [[buffer(1)]],
                                   constant uint &to [[buffer(2)]],
                                   device uint *out [[buffer(3)]],
                                   constant uint &pick [[buffer(4)]],
                                   uint i [[thread_position_in_grid]]) {
    device uint *dests[3] = {a, b, out};
    Destinations d = {array_ref<device uint *>(dests, 3), i};
    move_word(d, (pick + i) % 2, to, i);
}

kernel void pointer_array_zero_filled(device const uint *a [[buffer(0)]],
                                      device const uint *b [[buffer(1)]],
                                      device const uint *c [[buffer(2)]],
                                      device uint *out [[buffer(3)]],
                                      constant uint &count [[buffer(4)]],
                                      uint i [[thread_position_in_grid]]) {
    device const uint *srcs[16] = {};
    srcs[0] = a;
    if (count > 1)
        srcs[1] = b;
    if (count > 2)
        srcs[2] = c;
    uint n = min(count, 3u);
    for (uint k = 0; k < n; k++)
        srcs[k] += i;
    uint sum = 0;
    for (uint k = 0; k < n; k++)
        sum = sum * 31u + *srcs[k];
    out[i] = sum;
}
