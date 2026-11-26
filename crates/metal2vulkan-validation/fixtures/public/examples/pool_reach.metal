#include <metal_stdlib>
using namespace metal;

// Each thread reads the word `at[i]` words from `target`'s first word into `seen[i]`, then
// writes its mark there. The runtime's test places `target` between two neighbours in one block
// and gives indices inside it, before its first byte and past its end.
kernel void pool_reach(device uint *target [[buffer(0)]],
                       device uint *seen [[buffer(1)]],
                       constant int *at [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
    int k = at[i];
    seen[i] = target[k];
    target[k] = 0xa5a50000u | i;
}
