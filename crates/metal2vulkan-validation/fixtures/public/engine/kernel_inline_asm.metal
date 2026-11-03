// Owned fixture: inline assembly. An `asm volatile("")` statement in a loop body (the shape a
// shader generator writes so that the compiler keeps a loop) compiles to a call of
// `asm sideeffect "", ""`, a function-local INLINEASM constant (CONSTANTS_BLOCK record 30). MSL
// accepts only that statement: an empty template without operands or clobbers (any other is
// "illegal asm statement"). `asm_in_loop` and `asm_in_helper` call it where their entry reaches
// it; `asm_free` has none of its own, and translates although the module holds the others.
// Written for this repository; no third-party source.
#include <metal_stdlib>
using namespace metal;

// Each thread stores the xor of its index with every i below n.
kernel void asm_in_loop(device uint *o [[buffer(0)]],
                        constant uint &n [[buffer(1)]],
                        uint t [[thread_position_in_grid]]) {
    uint acc = 0;
    for (uint i = 0; i < n; i++) {
        __asm__ volatile("");
        acc += i ^ t;
    }
    o[t] = acc;
}

__attribute__((noinline)) static uint asm_helper(uint x) {
    __asm__ volatile("");
    return x * 3u + 1u;
}

kernel void asm_in_helper(device uint *o [[buffer(0)]], uint t [[thread_position_in_grid]]) {
    o[t] = asm_helper(t);
}

// The same store as asm_in_loop, without assembly.
kernel void asm_free(device uint *o [[buffer(0)]],
                     constant uint &n [[buffer(1)]],
                     uint t [[thread_position_in_grid]]) {
    uint acc = 0;
    for (uint i = 0; i < n; i++) {
        acc += i ^ t;
    }
    o[t] = acc;
}
