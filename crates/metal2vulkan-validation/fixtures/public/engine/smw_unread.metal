// Owned fixture: a writer of color 0 that declares color 0 as an input it never reads, and whose
// discard alone keeps it from early fragment tests. The function reads nothing of the attachment,
// so on a device that orders fetch through the pixel interlock its pipeline takes no storage form:
// the write it returns to color 0 is an ordinary color write, after the depth and stencil tests.
// It returns `tint` (buffer 0) and discards the fragments of every fourth column (`x` in
// `[4k, 4k + 1)`), as `smw_writers`' `smw_discard` does. Loop-free.
// metal-fast-math: off
// metal-macos-min: 27.0
#include <metal_stdlib>
using namespace metal;

fragment half4 smw_unread(float4 p [[position]], half4 unread [[color(0)]],
                          constant half4 &tint [[buffer(0)]]) {
    if ((uint(p.x) & 3u) == 0u) {
        discard_fragment();
    }
    return tint;
}
