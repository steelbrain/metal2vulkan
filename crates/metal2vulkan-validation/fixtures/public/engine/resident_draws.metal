// Owned fixture: the functions of the resident-draws lane (`runtime_lane::resident_draws`). A
// draw places one small triangle at a position: one its vertex buffer holds
// (`rd_slot_vertex`), or one the pointer of its argument-table entry points to
// (`rd_entry_vertex`). The fragment returns one colour (`rd_color_fragment`), or the texel at
// the centre of the texture its entry names, through the sampler its entry names
// (`rd_entry_fragment`), or counts itself into the word its tally's pointer names and returns one
// colour (`rd_tally_fragment`). An entry is 24 bytes: the position's address, the texture's
// resource ID and the sampler's; a tally is 8, the word's address. `rd_touch` counts its threads
// into a word. `rd_encode` counts too, and may reset render command 0 of the indirect command
// buffer its table names, past a count no frame reaches, so it writes no record. No function
// writes memory but `rd_tally_fragment`, `rd_touch` and `rd_encode`.
// Written for this repository; no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Dot {
    float4 position [[position]];
};

struct Entry {
    constant float4 *position;
    texture2d<float> texture;
    sampler texels;
};

static float2 corner(uint vid)
{
    return float2(vid == 1 ? 0.02 : 0.0, vid == 2 ? 0.02 : 0.0);
}

vertex Dot rd_slot_vertex(uint vid [[vertex_id]], constant float4 &at [[buffer(0)]])
{
    Dot d;
    d.position = float4(at.xy + corner(vid), 0.0, 1.0);
    return d;
}

fragment half4 rd_color_fragment()
{
    return half4(1.0, 0.5, 0.0, 1.0);
}

vertex Dot rd_entry_vertex(uint vid [[vertex_id]], constant Entry &entry [[buffer(0)]])
{
    Dot d;
    d.position = float4(entry.position->xy + corner(vid), 0.0, 1.0);
    return d;
}

fragment float4 rd_entry_fragment(constant Entry &entry [[buffer(0)]])
{
    return entry.texture.sample(entry.texels, float2(0.5, 0.5));
}

struct Tally {
    device atomic_uint *count;
};

fragment half4 rd_tally_fragment(constant Tally &tally [[buffer(0)]])
{
    atomic_fetch_add_explicit(tally.count, 1, memory_order_relaxed);
    return half4(1.0, 0.5, 0.0, 1.0);
}

kernel void rd_touch(device atomic_uint *count [[buffer(0)]])
{
    atomic_fetch_add_explicit(count, 1, memory_order_relaxed);
}

struct Commands {
    command_buffer commands;
};

kernel void rd_encode(device atomic_uint *count [[buffer(0)]],
                      device const Commands &table [[buffer(1)]])
{
    if (atomic_fetch_add_explicit(count, 1, memory_order_relaxed) == 0xffffffffu) {
        render_command command(table.commands, 0);
        command.reset();
    }
}
