// Owned fixture: a kernel's explicit imageblock whose cell pointer reaches past the layout's data
// size. In ib_half_cell the layout is `{half v}` (air.imageblock_data_size 2, align 2): each
// thread stores a half2 (tid.x, tid.y) through its cell pointer, and after the imageblock
// barrier thread (0, 0) writes the half2 slice at the cell's start to the texture at
// tg * 8. ib_half2_cell is its twin over a `{half2 v}` layout (data size 4). Metal runs both and
// writes each pixel its own (tid.x, tid.y); its imageblock is 8 bytes a pixel for either.
// Written for this repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct HalfCell {
    half v;
};

struct Half2Cell {
    half2 v;
};

kernel void ib_half_cell(imageblock<HalfCell, imageblock_layout_explicit> ib,
                         texture2d<half, access::write> t [[texture(0)]],
                         ushort2 tid [[thread_position_in_threadgroup]],
                         uint2 tg [[threadgroup_position_in_grid]]) {
    threadgroup_imageblock HalfCell *c = ib.data(tid);
    *(threadgroup_imageblock half2 *)c = half2(half(tid.x), half(tid.y));
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    if (all(tid == 0)) {
        t.write(ib.slice(*(threadgroup_imageblock half2 *)&c->v), ushort2(tg * 8));
    }
}

kernel void ib_half2_cell(imageblock<Half2Cell, imageblock_layout_explicit> ib,
                          texture2d<half, access::write> t [[texture(0)]],
                          ushort2 tid [[thread_position_in_threadgroup]],
                          uint2 tg [[threadgroup_position_in_grid]]) {
    threadgroup_imageblock Half2Cell *c = ib.data(tid);
    c->v = half2(half(tid.x), half(tid.y));
    threadgroup_barrier(mem_flags::mem_threadgroup_imageblock);
    if (all(tid == 0)) {
        t.write(ib.slice(c->v), ushort2(tg * 8));
    }
}
