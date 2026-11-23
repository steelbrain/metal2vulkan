// Owned MSL: visible functions of the type `uint(uint, uint)` that a kernel's visible-function
// table holds. A case links them (by this module's `.ll` hash) into `visible_table_dispatch` of
// `engine/visible_table_caller.metal`, which calls entry `tid % size` with `(tid, 3)`. Loop-free.
#include <metal_stdlib>
using namespace metal;

[[visible]] uint table_add(uint a, uint b) { return a + b; }

[[visible]] uint table_mul(uint a, uint b) { return a * b; }

[[visible]] uint table_sub(uint a, uint b) { return a - b; }
