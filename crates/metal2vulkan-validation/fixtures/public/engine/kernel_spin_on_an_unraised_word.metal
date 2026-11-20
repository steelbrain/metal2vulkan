// Owned fixture: the busy work of a queue storm. One
// thread loads a word no one raises, with relaxed atomic loads, until it has made `cap` of them,
// then stores how many it made: its GPU time grows with `cap` alone. It calls nothing a translation
// drops (rfs_agx_yield's `llvm.agx3.yield` delays only Metal's own compilation of a spin), so
// Metal's compilation and the translation's spin the same loads.
#include <metal_stdlib>
using namespace metal;

kernel void spin_on_an_unraised_word(device atomic_uint *word [[buffer(0)]],
                                     device uint *loads [[buffer(1)]],
                                     constant uint &cap [[buffer(2)]]) {
    uint k = 0;
    while (k < cap) {
        if (atomic_load_explicit(word, memory_order_relaxed) != 0u) {
            break;
        }
        k++;
    }
    loads[0] = k;
}
