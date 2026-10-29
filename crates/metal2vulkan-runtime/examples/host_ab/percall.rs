//! The per-call cost table (`--only percall`): one group a call, each call timed alone on both
//! sides, with Metal's cost as the budget.
//!
//! A group's window prepares the state its call needs, untimed; makes that one call `inner`
//! times; then drains the state, committing and waiting where the calls encoded work. Three
//! warm windows run first, and every pipeline state, library and texture the groups bind is made
//! before any of them, so no making falls inside a timed window (the `variant_pipelines_made`
//! count a group reports is the proof).
//!
//! Per call a group reports
//!
//! - `<call>_self`: the calling thread's wall time, the cost the caller pays inline;
//! - `<call>_blocked`: how much of that the calling thread was off a CPU, so it waited;
//! - `<call>_others`: our own other threads' CPU time while the calls ran;
//! - `<call>_drain`: the calling thread's wall time in the drain after the calls, where the
//!   group has one (its command buffer's commit and wait): what the caller pays later;
//! - `<call>_drain_blocked`: how much of that drain the calling thread was off a CPU, so a
//!   drain that costs CPU instead was spent spinning;
//! - `<call>_drain_others`: our other threads' CPU time in that drain;
//! - `<call>_allocs`: the heap allocations a call made on the calling thread;
//! - `<call>_rt_<count>`: the runtime's own counts of what the calls made, where they moved
//!   ([`counted`]); none natively, so these rows have no ratio. Every count the runtime keeps is
//!   reported where a call moved it, and `_at_call` and `_at_commit` say whether the call paid
//!   for it or left it for the commit — including the driver's own makings, which the runtime
//!   counts where it calls them (`churn_counts.rs`), so a creation call's driver share is a count
//!   and not an inference from the driver floor below the table.
//!
//! The wall times are a median over the windows, and the rest the group's totals over its
//! calls, because the thread and process CPU clocks answer in microseconds.
//!
//! The counts are also the table's **gate**: [`HELD`] holds named calls to what they were
//! measured making, and the run fails where one makes more. A count is deterministic, so it holds
//! in CI under any load, where none of the times above can.

use crate::gpu::{Format, Gpu};
use crate::workloads::{emit, host_pages, process_cpu, thread_cpu, Scene, Sizes, TARGET_SIDE};
use std::time::{Duration, Instant};

/// The windows a group runs before the measured ones.
const WARM: usize = 3;

/// Whether a count of the side's own ([`Gpu::work_counts`]) belongs in the table: every count
/// of work does, and the accumulators of nanoseconds do not, since a time is not a fact a gate
/// can hold. Every other count the side keeps is reported where a group's calls moved it, so no
/// family of work is left out of a call's row by an allowlist nobody revisits.
fn counted(name: &str) -> bool {
    !name.ends_with("_nanos")
}

/// What each group's call is on the runtime's public API, one entry a group: the group's name,
/// then the methods it calls, `Type::method`. [`group`] fails for a group that is not named here
/// (a driver call of `vkfloor.rs`, whose name starts with `vk_`, is no Metal call and is
/// exempt), and `scripts/api-cost-coverage` reads it to say how much of the public surface the
/// table times and how much is still untimed.
pub const TIMED: [(&str, &str); 60] = [
    ("command_buffer", "CommandQueue::command_buffer"),
    ("commit_empty", "CommandBuffer::commit"),
    ("commit_one_dispatch", "CommandBuffer::commit"),
    ("on_completed", "CommandBuffer::add_completed_handler"),
    ("status_completed", "CommandBuffer::status"),
    (
        "times_completed",
        "CommandBuffer::kernel_start_time, CommandBuffer::kernel_end_time, \
         CommandBuffer::gpu_start_time, CommandBuffer::gpu_end_time",
    ),
    ("wait_completed", "CommandBuffer::wait_until_completed"),
    (
        "render_encoder_pass",
        "CommandBuffer::render_command_encoder",
    ),
    (
        "render_encoder_clear",
        "CommandBuffer::render_command_encoder",
    ),
    (
        "render_encoder_load",
        "CommandBuffer::render_command_encoder",
    ),
    ("end_render", "RenderCommandEncoder::end_encoding"),
    ("compute_encoder", "CommandBuffer::compute_command_encoder"),
    ("end_compute", "ComputeCommandEncoder::end_encoding"),
    ("blit_encoder", "CommandBuffer::blit_command_encoder"),
    ("end_blit", "BlitCommandEncoder::end_encoding"),
    (
        "set_render_pipeline",
        "RenderCommandEncoder::set_render_pipeline_state",
    ),
    (
        "set_vertex_buffer",
        "RenderCommandEncoder::set_vertex_buffer",
    ),
    (
        "set_vertex_buffers_two",
        "RenderCommandEncoder::set_vertex_buffers",
    ),
    (
        "set_vertex_bytes_16",
        "RenderCommandEncoder::set_vertex_bytes",
    ),
    (
        "set_fragment_buffer",
        "RenderCommandEncoder::set_fragment_buffer",
    ),
    (
        "set_fragment_texture",
        "RenderCommandEncoder::set_fragment_texture",
    ),
    (
        "set_fragment_sampler",
        "RenderCommandEncoder::set_fragment_sampler_state",
    ),
    ("set_cull_mode", "RenderCommandEncoder::set_cull_mode"),
    (
        "set_depth_stencil",
        "RenderCommandEncoder::set_depth_stencil_state",
    ),
    ("draw_strip", "RenderCommandEncoder::draw_primitives"),
    ("draw_strip_in_64", "RenderCommandEncoder::draw_primitives"),
    ("draw_triangles", "RenderCommandEncoder::draw_primitives"),
    (
        "draw_indexed",
        "RenderCommandEncoder::draw_indexed_primitives",
    ),
    (
        "use_resources_read_1_drawn",
        "RenderCommandEncoder::use_resources",
    ),
    (
        "use_resources_read_256_drawn",
        "RenderCommandEncoder::use_resources",
    ),
    (
        "set_compute_pipeline",
        "ComputeCommandEncoder::set_compute_pipeline_state",
    ),
    ("set_buffer", "ComputeCommandEncoder::set_buffer"),
    ("set_texture", "ComputeCommandEncoder::set_texture"),
    ("dispatch", "ComputeCommandEncoder::dispatch_threadgroups"),
    (
        "use_resources_read_1",
        "ComputeCommandEncoder::use_resources",
    ),
    (
        "use_resources_read_256",
        "ComputeCommandEncoder::use_resources",
    ),
    (
        "use_resource_read_write",
        "ComputeCommandEncoder::use_resource",
    ),
    ("use_heap", "ComputeCommandEncoder::use_heap"),
    ("update_fence", "ComputeCommandEncoder::update_fence"),
    ("wait_for_fence", "ComputeCommandEncoder::wait_for_fence"),
    (
        "copy_buffer_1k",
        "BlitCommandEncoder::copy_from_buffer_to_buffer",
    ),
    ("fill_buffer_256", "BlitCommandEncoder::fill_buffer"),
    (
        "copy_buffer_to_texture",
        "BlitCommandEncoder::copy_from_buffer_to_texture",
    ),
    (
        "fill_in_own_encoder",
        "CommandBuffer::blit_command_encoder, BlitCommandEncoder::fill_buffer, \
         BlitCommandEncoder::end_encoding",
    ),
    ("buffer_4k", "Device::new_buffer"),
    ("buffer_1m", "Device::new_buffer"),
    ("buffer_with_bytes_4k", "Device::new_buffer_with_bytes"),
    ("buffer_no_copy_1m", "Device::new_buffer_with_bytes_no_copy"),
    ("texture_256_shared", "Device::new_texture"),
    ("texture_1024_target", "Device::new_texture"),
    ("heap_texture_256", "Heap::new_texture_at_offset"),
    ("buffer_texture_256", "Buffer::new_texture"),
    ("sampler", "Device::new_sampler_state"),
    ("shared_event", "Device::new_shared_event"),
    ("fence", "Device::new_fence"),
    ("event_value", "SharedEvent::signaled_value"),
    ("set_event_value", "SharedEvent::set_signaled_value"),
    ("read_buffer_contents_256", "Buffer::contents"),
    ("get_bytes_256", "Texture::get_bytes"),
    ("replace_region_256", "Texture::replace_region"),
];

/// The counts [`counted`] keeps that moved between two readings of the side's own counts, and by
/// how much: none at all on a side that keeps none.
fn moved(from: &[(&'static str, f64)], to: &[(&'static str, f64)]) -> Vec<(&'static str, f64)> {
    from.iter()
        .filter(|(name, _)| counted(name))
        .filter_map(|(counted, before)| {
            let now = to
                .iter()
                .find(|(name, _)| name == counted)
                .map_or(0.0, |(_, count)| *count);
            (now > *before).then_some((*counted, now - before))
        })
        .collect()
}

/// What one window measured: the calls' wall time on the calling thread, that thread's CPU time
/// and the whole process's while they ran, then the same three of the drain after them.
#[derive(Clone, Copy, Default)]
struct Sample {
    wall: Duration,
    cpu: Duration,
    process: Duration,
    drain_wall: Duration,
    drain_cpu: Duration,
    drain_process: Duration,
}

/// A group's windows, summed.
#[derive(Default)]
struct Totals {
    /// Each window's wall time a call, in ns: the call's own cost.
    own: Vec<f64>,
    /// Each window's drain, in ns a call.
    drain: Vec<f64>,
    sums: Sample,
}

impl Totals {
    fn add(&mut self, inner: usize, s: Sample) {
        let each = |d: Duration| d.as_nanos() as f64 / inner as f64;
        self.own.push(each(s.wall));
        self.drain.push(each(s.drain_wall));
        let sums = &mut self.sums;
        sums.wall += s.wall;
        sums.cpu += s.cpu;
        sums.process += s.process;
        sums.drain_wall += s.drain_wall;
        sums.drain_cpu += s.drain_cpu;
        sums.drain_process += s.drain_process;
    }
}

/// Time one call group: `windows` measured windows, each preparing the call's state, making the
/// call `inner` times, then draining it. Every window runs inside an autorelease pool, as a
/// Metal program's frame does, so nothing of one window is released inside another's timing.
pub fn group<G: Gpu, S>(
    g: &G,
    name: &str,
    (inner, windows): (usize, usize),
    mut prepare: impl FnMut() -> S,
    mut call: impl FnMut(&mut S, usize),
    mut drain: impl FnMut(S),
) {
    assert!(inner > 0 && windows > 0, "{name}: no calls");
    assert!(
        name.starts_with("vk_") || TIMED.iter().any(|(group, _)| *group == name),
        "{name}: no entry in TIMED, so the timing census would not count its calls"
    );
    let mut totals = Totals::default();
    let mut before = vec![];
    for window in 0..WARM + windows {
        if window == WARM {
            before = g.work_counts();
        }
        let sample = g.pool(|| {
            let mut state = prepare();
            let (cpu0, process0) = (thread_cpu(), process_cpu());
            let start = Instant::now();
            for i in 0..inner {
                call(&mut state, i);
            }
            let called = Instant::now();
            let (cpu1, process1) = (thread_cpu(), process_cpu());
            let drained_from = Instant::now();
            drain(state);
            let drained = Instant::now();
            let (cpu2, process2) = (thread_cpu(), process_cpu());
            Sample {
                wall: called - start,
                cpu: cpu1 - cpu0,
                process: process1 - process0,
                drain_wall: drained - drained_from,
                drain_cpu: cpu2 - cpu1,
                drain_process: process2 - process1,
            }
        });
        if window >= WARM {
            totals.add(inner, sample);
        }
    }
    let after = g.work_counts();
    let calls = (inner * windows) as f64;
    let each = |d: Duration| d.as_nanos() as f64 / calls;
    let sums = &totals.sums;
    let metric = |what: &str| format!("{name}_{what}");
    emit("percall", &metric("self"), "ns/call", &mut totals.own);
    emit(
        "percall",
        &metric("blocked"),
        "ns/call",
        &mut [(each(sums.wall) - each(sums.cpu)).max(0.0)],
    );
    emit(
        "percall",
        &metric("others"),
        "ns/call",
        &mut [(each(sums.process) - each(sums.cpu)).max(0.0)],
    );
    emit("percall", &metric("drain"), "ns/call", &mut totals.drain);
    emit(
        "percall",
        &metric("drain_blocked"),
        "ns/call",
        &mut [(each(sums.drain_wall) - each(sums.drain_cpu)).max(0.0)],
    );
    emit(
        "percall",
        &metric("drain_others"),
        "ns/call",
        &mut [(each(sums.drain_process) - each(sums.drain_cpu)).max(0.0)],
    );
    // One window more, this thread's allocations counted, and the runtime's counts read between
    // the calls and the drain as well: what a call made the runtime do at the call, and what it
    // left for the commit.
    let (made, at_call, at_drain) = g.pool(|| {
        let mut state = prepare();
        let opened = g.work_counts();
        let made = crate::allocs::counting(|| {
            let before = crate::allocs::this_thread();
            for i in 0..inner {
                call(&mut state, i);
            }
            crate::allocs::this_thread() - before
        });
        let called = g.work_counts();
        drain(state);
        let drained = g.work_counts();
        (made, moved(&opened, &called), moved(&called, &drained))
    });
    // The counts this group measured, under the names the table prints, so [`HELD`]'s rows for it
    // are checked against the same numbers the report shows.
    let mut measured: Vec<(String, f64)> = vec![];
    let mut keep = |what: String, value: f64| {
        emit(
            "percall",
            &metric(&what),
            if what == "allocs" {
                "allocs/call"
            } else {
                "count/call"
            },
            &mut [value],
        );
        measured.push((what, value));
    };
    keep("allocs".to_owned(), made as f64 / inner as f64);
    for (counted, count) in at_call {
        keep(format!("rt_{counted}_at_call"), count / inner as f64);
    }
    for (counted, count) in at_drain {
        keep(format!("rt_{counted}_at_commit"), count / inner as f64);
    }
    // What the measured windows' calls made the runtime do, per call.
    for (which, from) in &before {
        if !counted(which) {
            continue;
        }
        let now = after
            .iter()
            .find(|(counted, _)| counted == which)
            .map_or(0.0, |(_, count)| *count);
        assert!(
            *which != "variant_pipelines_made" || now <= *from,
            "{name}: {} pipeline states were made inside its timed windows",
            now - from
        );
        if now > *from {
            keep(format!("rt_{which}"), (now - from) / calls);
        }
    }
    // The side that keeps no counts of its own is Metal's, which has no counterpart for any of
    // these, so only ours is held to them.
    if !before.is_empty() {
        hold(name, &measured);
    }
}

/// What a call must have made, per call: the group, the count under the name the table prints
/// (`allocs`, or `rt_<count>_at_call` and `rt_<count>_at_commit` for the runtime's own counts),
/// and the most a call of it may make. A count a group never moved is not printed, and reads as
/// zero here, so a row of `0.0` holds a call to making none of that work at all.
///
/// These are the gate on the cost half. A count is the same under any load and on any host, so a
/// regression reads "this call now makes two allocations where it made none" and fails the run,
/// where a 12% change in a time would not. Each row is a fact measured on this M3 Max that the
/// runtime should not quietly lose: an upper bound, so a call that does *less* work passes and the
/// row is then tightened.
const HELD: [(&str, &str, f64); 43] = [
    // A draw records exactly one draw command and makes no driver object: the cost it pays at the
    // call is planning, which is settled work and nothing a driver charges for.
    ("draw_strip", "rt_draws_at_commit", 1.0),
    ("draw_strip", "rt_draws_at_call", 0.0),
    ("draw_strip", "rt_vk_images_made_at_call", 0.0),
    ("draw_strip", "rt_vk_buffers_made_at_call", 0.0),
    ("draw_strip", "rt_vk_memory_allocations_at_call", 0.0),
    ("draw_triangles", "rt_draws_at_commit", 1.0),
    (
        "draw_indexed",
        "rt_recorded_cmd_draw_indexed_at_commit",
        1.0,
    ),
    // And it allocates about once in twenty draws (0.049 measured), nothing a draw of its own.
    ("draw_strip", "allocs", 0.2),
    ("draw_triangles", "allocs", 0.2),
    ("draw_indexed", "allocs", 0.2),
    // The draw's settling is three lookups that each find what an earlier draw made: a draw that
    // starts recording shapes or sets again would show here before it showed in a time.
    ("draw_strip", "rt_draw_shapes_made_at_call", 0.0),
    ("draw_strip", "rt_sampler_sets_made_at_call", 0.0),
    ("draw_strip", "rt_live_tables_uploaded_at_call", 0.0),
    // The render pass is begun at the commit, never at the call, and ending the encoding makes no
    // driver object. Its 25 allocations are the largest counted quantity on the path.
    ("end_render", "rt_render_passes_begun_at_call", 0.0),
    ("end_render", "rt_vk_images_made_at_call", 0.0),
    ("end_render", "allocs", 25.0),
    ("end_compute", "allocs", 7.0),
    ("end_blit", "allocs", 7.0),
    ("dispatch", "rt_dispatches_at_commit", 1.0),
    ("dispatch", "allocs", 19.0),
    // A heap texture's image takes its storage from the heap: one image and one binding, and no
    // allocation of memory of its own.
    ("heap_texture_256", "rt_vk_images_made_at_call", 1.0),
    ("heap_texture_256", "rt_vk_image_bindings_at_call", 1.0),
    ("heap_texture_256", "rt_vk_memory_allocations_at_call", 0.0),
    ("heap_texture_256", "allocs", 19.0),
    // A Shared texture likewise, and it starts its image on the host once.
    ("texture_256_shared", "rt_vk_images_made_at_call", 1.0),
    ("texture_256_shared", "rt_vk_image_bindings_at_call", 1.0),
    (
        "texture_256_shared",
        "rt_vk_memory_allocations_at_call",
        0.0,
    ),
    (
        "texture_256_shared",
        "rt_images_started_on_the_host_at_call",
        1.0,
    ),
    ("texture_256_shared", "allocs", 32.0),
    // A render target of a shape the runtime has seen takes a parked image: it makes no image at
    // all, which is why it costs what Metal's does. Losing the recycler here would cost 20 us a
    // call and this row is what would say so.
    ("texture_1024_target", "rt_vk_images_made_at_call", 0.0),
    (
        "texture_1024_target",
        "rt_vk_memory_allocations_at_call",
        0.0,
    ),
    // A texture over a buffer's bytes takes a parked image too, and makes one view of it.
    ("buffer_texture_256", "rt_vk_images_made_at_call", 0.0),
    ("buffer_texture_256", "rt_vk_image_views_made_at_call", 1.0),
    // A buffer comes out of the runtime's memory pool: no `vkCreateBuffer`, no `vkAllocateMemory`.
    ("buffer_4k", "rt_vk_buffers_made_at_call", 0.0),
    ("buffer_4k", "rt_vk_memory_allocations_at_call", 0.0),
    ("buffer_1m", "rt_vk_buffers_made_at_call", 0.0),
    ("buffer_1m", "rt_vk_memory_allocations_at_call", 0.0),
    // A no-copy buffer imports its pages once: one buffer, one import, one binding. A walk of the
    // memory types that tried several would read as more than one allocation here.
    ("buffer_no_copy_1m", "rt_vk_buffers_made_at_call", 1.0),
    ("buffer_no_copy_1m", "rt_vk_memory_allocations_at_call", 1.0),
    ("buffer_no_copy_1m", "rt_vk_buffer_bindings_at_call", 1.0),
    // Residency is our own bookkeeping: no driver call, and the allocation a resource the standing
    // cost of 256 of them.
    ("use_resources_read_256", "rt_vk_buffers_made_at_call", 0.0),
    ("use_resources_read_256", "allocs", 277.0),
    // One Metal command buffer carrying one dispatch is submitted as two of Vulkan's.
    (
        "commit_one_dispatch",
        "rt_vk_command_buffers_submitted",
        2.0,
    ),
];

/// Fail where a group's call made more work than [`HELD`] allows it, naming the call, the count,
/// what it made and what it is held to.
///
/// A count a group never moved is not printed, so a row that names one reads zero and would pass
/// whatever the call did: a row holding a call to *some* work must therefore name a count the
/// group moved, or it is a misspelling that holds nothing, and that fails too.
fn hold(name: &str, measured: &[(String, f64)]) {
    for (group, what, most) in HELD {
        if group != name {
            continue;
        }
        let made = measured.iter().find(|(counted, _)| counted == what);
        assert!(
            most <= 0.0 || made.is_some(),
            "{name}: {what} is held to at most {most}, but the group moved no such count, so the \
             row holds nothing"
        );
        let made = made.map_or(0.0, |(_, value)| *value);
        assert!(
            made <= most,
            "{name}: {what} is {made} a call, where it is held to at most {most}"
        );
    }
}

/// How many bytes of host pages the no-copy buffers of one window lie over, and how many such
/// buffers a window makes: one megabyte each, the same pages every window, as a guest's buffers
/// lie in its memory.
const NO_COPY_EACH: usize = 1 << 20;
const NO_COPY_BUFFERS: usize = 8;

/// The buffers a residency group makes resident, 256 bytes each.
const RESIDENT: usize = 256;

/// The side of the textures the creation and upload groups make.
const SMALL_SIDE: usize = 256;

/// The bytes a small texture's level holds, as `BGRA8Unorm` of unpadded rows.
const SMALL_BYTES: usize = SMALL_SIDE * SMALL_SIDE * 4;

/// The textures a placement heap holds in the heap-texture and `useHeap` groups.
const HEAP_TEXTURES: usize = 16;

/// What the groups bind, draw and copy: made once, before any window.
struct Objects<G: Gpu> {
    /// A render pass of the scene's target, cleared and stored, made once for many encoders.
    pass: G::Pass,
    /// `calls_vertex` with the texture-reading and the buffer-reading fragment function.
    textured: G::RenderPipeline,
    buffered: G::RenderPipeline,
    /// `quad_vertex` + `textured_fragment` into the depth pass's attachments.
    depth_state: G::RenderPipeline,
    depth: G::Texture,
    /// A depth compare with writes, and one without.
    depth_stencils: [G::DepthStencil; 2],
    /// The vertex buffers `calls_vertex` reads, and the fragment buffer `calls_buffer_fragment`
    /// does.
    positions: G::Buffer,
    colours: G::Buffer,
    uniforms: G::Buffer,
    /// Sixteen 16-bit indices of a triangle.
    indices: G::Buffer,
    /// A compute encoder's bound buffers, and a blit's source and destination.
    input: G::Buffer,
    output: G::Buffer,
    from: G::Buffer,
    to: G::Buffer,
    /// A small texture's bytes, for the upload and `replaceRegion` groups, and a texture of that
    /// size to upload into and read back.
    texels: Vec<u8>,
    small: G::Buffer,
    small_texture: G::Texture,
    /// The buffers the residency groups make resident.
    resident: Vec<G::Buffer>,
    /// A placement heap of `Shared` storage, whose hazards nothing tracks: what `useHeap` is
    /// for, and where the heap-texture group places its textures.
    heap: G::Heap,
    /// A shared event, a fence, and the pages the no-copy buffers lie over.
    event: G::SharedEvent,
    fence: G::Fence,
    pages: *mut u8,
    /// A command buffer of one fill, committed and completed, whose status, times and wait the
    /// read groups take.
    done: G::CommandBuffer,
}

impl<G: Gpu> Objects<G> {
    fn new(g: &G, s: &Scene<G>, library: &G::Library) -> Objects<G> {
        let state =
            |fragment| g.render_pipeline(library, "calls_vertex", fragment, Format::Bgra8, false);
        let floats =
            |values: &[f32]| -> Vec<u8> { values.iter().flat_map(|v| v.to_le_bytes()).collect() };
        // A strip of four corners of a small square, then nothing, as the calls workload's.
        let positions: Vec<f32> = (0..256)
            .flat_map(|v| match v {
                0..4 => [
                    -0.1 + 0.2 * (v & 1) as f32,
                    -0.1 + 0.2 * (v >> 1) as f32,
                    0.0,
                    1.0,
                ],
                _ => [0.0; 4],
            })
            .collect();
        let texels: Vec<u8> = (0..SMALL_BYTES).map(|i| (i % 251) as u8).collect();
        let done = g.command_buffer();
        g.fill(&done, &s.fill, 256);
        g.commit(&done);
        g.wait(&done);
        Objects {
            pass: g.pass(&s.target),
            textured: state("calls_texture_fragment"),
            buffered: state("calls_buffer_fragment"),
            depth_state: g.depth_pipeline(library, "quad_vertex", "textured_fragment", false),
            depth: g.texture(Format::Depth32Float, TARGET_SIDE, TARGET_SIDE, true),
            depth_stencils: [g.depth_stencil(true), g.depth_stencil(false)],
            positions: g.buffer_with(&floats(&positions)),
            colours: g.buffer_with(&floats(&[0.5f32, 0.25, 0.75, 1.0].repeat(256))),
            uniforms: g.buffer_with(&floats(&[0.25f32, 0.5, 0.75, 1.0].repeat(4096))),
            indices: g.buffer_with(&(0..16u16).flat_map(u16::to_le_bytes).collect::<Vec<u8>>()),
            input: g.buffer_with(&vec![1u8; 1 << 16]),
            output: g.buffer(1 << 16),
            from: g.buffer_with(&vec![0x3cu8; 1 << 18]),
            to: g.buffer(1 << 18),
            small: g.buffer_with(&texels),
            small_texture: g.texture(Format::Bgra8, SMALL_SIDE, SMALL_SIDE, false),
            texels,
            resident: (0..RESIDENT).map(|_| g.buffer(256)).collect(),
            heap: g.heap(HEAP_TEXTURES * SMALL_BYTES * 2),
            event: g.shared_event(),
            fence: g.fence(),
            // SAFETY: the pages outlive every buffer made over them; nothing unmaps them.
            pages: unsafe { host_pages(NO_COPY_BUFFERS * NO_COPY_EACH) },
            done,
        }
    }
}

/// The per-call table: every group, in the order a frame makes its calls.
pub fn percall<G: Gpu>(g: &G, s: &Scene<G>, library: &G::Library, sizes: &Sizes) {
    let o = Objects::new(g, s, library);
    let w = sizes.timed;
    // A group whose window makes one call, because a second call of it would not pay what the
    // first did, runs more windows: its samples are its windows, and the host clock this Mac
    // counts steps in 41.67 ns, so a call of a few tens of nanoseconds needs many of them.
    let lone = w * 6;
    let made_before = g.work_counts();

    // The command buffer: made, committed empty, committed with one encoder's work, and the
    // status, times and completed wait a frame's end reads.
    group(
        g,
        "command_buffer",
        (32, w),
        || Vec::with_capacity(32),
        |held: &mut Vec<G::CommandBuffer>, _| held.push(g.command_buffer()),
        drop,
    );
    group(
        g,
        "commit_empty",
        (16, w),
        || (0..16).map(|_| g.command_buffer()).collect::<Vec<_>>(),
        |held: &mut Vec<G::CommandBuffer>, i| g.commit(&held[i]),
        |held| held.iter().for_each(|cb| g.wait(cb)),
    );
    group(
        g,
        "commit_one_dispatch",
        (16, w),
        || {
            (0..16)
                .map(|_| {
                    let cb = g.command_buffer();
                    let mut e = g.compute_encoder(&cb);
                    g.set_compute_pipeline(&mut e, &s.tiny[0]);
                    g.set_buffer(&mut e, &o.output, 0, 0);
                    g.set_buffer(&mut e, &o.input, 0, 1);
                    g.dispatch(&mut e, 1, 64);
                    g.end_compute(e);
                    cb
                })
                .collect::<Vec<_>>()
        },
        |held: &mut Vec<G::CommandBuffer>, i| g.commit(&held[i]),
        |held| held.iter().for_each(|cb| g.wait(cb)),
    );
    group(
        g,
        "on_completed",
        (16, w),
        || (0..16).map(|_| g.command_buffer()).collect::<Vec<_>>(),
        |held: &mut Vec<G::CommandBuffer>, i| g.on_completed(&held[i], Box::new(|| {})),
        |held| {
            held.iter().for_each(|cb| g.commit(cb));
            held.iter().for_each(|cb| g.wait(cb));
        },
    );
    assert_eq!(g.status(&o.done), 4, "the read groups' command buffer");
    group(
        g,
        "status_completed",
        (256, w),
        || (),
        |_, _| assert_eq!(std::hint::black_box(g.status(&o.done)), 4),
        drop,
    );
    group(
        g,
        "times_completed",
        (256, w),
        || (),
        |_, _| {
            std::hint::black_box(g.times(&o.done));
        },
        drop,
    );
    group(
        g,
        "wait_completed",
        (256, w),
        || (),
        |_, _| g.wait_only(&o.done),
        drop,
    );

    // The encoders: opened and ended. A window opens or ends one encoder, so neither call pays
    // for the other, and the command buffer the window built is committed and waited in its
    // drain, so the work our threads do for it lands in the drain's columns.
    group(
        g,
        "render_encoder_pass",
        (1, lone),
        || g.command_buffer(),
        |cb: &mut G::CommandBuffer, _| {
            let e = g.pass_encoder(cb, &o.pass);
            g.end_render(e);
        },
        |cb| {
            g.commit(&cb);
            g.wait(&cb);
        },
    );
    for (name, clear) in [
        ("render_encoder_clear", true),
        ("render_encoder_load", false),
    ] {
        group(
            g,
            name,
            (1, lone),
            || g.command_buffer(),
            |cb: &mut G::CommandBuffer, _| {
                let e = g.render_encoder(cb, &s.target, clear);
                g.end_render(e);
            },
            |cb| {
                g.commit(&cb);
                g.wait(&cb);
            },
        );
    }
    group(
        g,
        "end_render",
        (1, lone),
        || {
            let cb = g.command_buffer();
            let e = g.pass_encoder(&cb, &o.pass);
            (cb, Some(e))
        },
        |state: &mut (G::CommandBuffer, Option<G::RenderEncoder>), _| {
            g.end_render(state.1.take().expect("the window's encoder"))
        },
        |(cb, _)| {
            g.commit(&cb);
            g.wait(&cb);
        },
    );
    group(
        g,
        "compute_encoder",
        (1, lone),
        || g.command_buffer(),
        |cb: &mut G::CommandBuffer, _| {
            let e = g.compute_encoder(cb);
            g.end_compute(e);
        },
        |cb| {
            g.commit(&cb);
            g.wait(&cb);
        },
    );
    group(
        g,
        "end_compute",
        (1, lone),
        || {
            let cb = g.command_buffer();
            let mut e = g.compute_encoder(&cb);
            g.set_compute_pipeline(&mut e, &s.tiny[0]);
            g.set_buffer(&mut e, &o.output, 0, 0);
            g.set_buffer(&mut e, &o.input, 0, 1);
            g.dispatch(&mut e, 1, 64);
            (cb, Some(e))
        },
        |state: &mut (G::CommandBuffer, Option<G::ComputeEncoder>), _| {
            g.end_compute(state.1.take().expect("the window's encoder"))
        },
        |(cb, _)| {
            g.commit(&cb);
            g.wait(&cb);
        },
    );
    group(
        g,
        "blit_encoder",
        (1, lone),
        || g.command_buffer(),
        |cb: &mut G::CommandBuffer, _| {
            let e = g.blit_encoder(cb);
            g.end_blit(e);
        },
        |cb| {
            g.commit(&cb);
            g.wait(&cb);
        },
    );
    group(
        g,
        "end_blit",
        (1, lone),
        || {
            let cb = g.command_buffer();
            let mut e = g.blit_encoder(&cb);
            g.copy_buffer(&mut e, (&o.from, 0), (&o.to, 0), 1024);
            (cb, Some(e))
        },
        |state: &mut (G::CommandBuffer, Option<G::BlitEncoder>), _| {
            g.end_blit(state.1.take().expect("the window's encoder"))
        },
        |(cb, _)| {
            g.commit(&cb);
            g.wait(&cb);
        },
    );

    // The state a draw binds. Each window's encoder is committed and waited, so the binds' later
    // cost shows in the drain.
    let render = |name: &str, binds: usize, bind: &dyn Fn(&mut G::RenderEncoder, usize)| {
        group(
            g,
            name,
            (binds, w),
            || {
                let cb = g.command_buffer();
                let mut e = g.pass_encoder(&cb, &o.pass);
                g.set_render_pipeline(&mut e, &o.textured);
                g.set_vertex_buffers(&mut e, &[&o.positions, &o.colours], &[0, 0], 0);
                g.set_fragment_texture(&mut e, &s.textures[0], 0);
                g.set_fragment_sampler(&mut e, &s.sampler, 0);
                (cb, Some(e))
            },
            |state: &mut (G::CommandBuffer, Option<G::RenderEncoder>), i| {
                bind(state.1.as_mut().expect("the window's encoder"), i)
            },
            |(cb, e)| {
                g.end_render(e.expect("the window's encoder"));
                g.commit(&cb);
                g.wait(&cb);
            },
        );
    };
    render("set_render_pipeline", 256, &|e, i| {
        g.set_render_pipeline(e, [&o.textured, &o.buffered][i % 2])
    });
    render("set_vertex_buffer", 256, &|e, i| {
        g.set_vertex_buffer(e, &o.positions, (i % 64) * 16, 0)
    });
    render("set_vertex_buffers_two", 256, &|e, i| {
        g.set_vertex_buffers(e, &[&o.positions, &o.colours], &[0, (i % 64) * 16], 0)
    });
    const SIXTEEN: [u8; 16] = [0x3c; 16];
    render("set_vertex_bytes_16", 256, &|e, _| {
        g.set_vertex_bytes(e, &SIXTEEN, 2)
    });
    render("set_fragment_buffer", 256, &|e, i| {
        g.set_fragment_buffer(e, &o.uniforms, (i % 256) * 256, 0)
    });
    render("set_fragment_texture", 256, &|e, i| {
        g.set_fragment_texture(e, &s.textures[i % s.textures.len()], 0)
    });
    render("set_fragment_sampler", 256, &|e, _| {
        g.set_fragment_sampler(e, &s.sampler, 0)
    });
    render("set_cull_mode", 256, &|e, i| g.set_cull_back(e, i % 2 == 0));
    let resident: Vec<&G::Buffer> = o.resident.iter().collect();
    render("use_resources_read_256_drawn", 1, &|e, _| {
        g.use_buffers_drawn(e, &resident)
    });
    render("use_resources_read_1_drawn", 1, &|e, _| {
        g.use_buffers_drawn(e, &[&o.resident[0]])
    });

    // The draws, each of a strip of four vertices, and the state they draw under set once. The
    // short encoder's draws cost what the long one's do, or the cost is not a call's but grows
    // with the encoder's length.
    render("draw_strip", 512, &|e, _| g.draw_strip(e, 4));
    render("draw_strip_in_64", 64, &|e, _| g.draw_strip(e, 4));
    render("draw_triangles", 512, &|e, _| g.draw_triangles(e, 0, 3));
    render("draw_indexed", 512, &|e, _| {
        g.draw_indexed(e, 3, &o.indices)
    });

    // A depth pass: the depth-stencil state, which no other group binds.
    group(
        g,
        "set_depth_stencil",
        (256, w),
        || {
            let cb = g.command_buffer();
            let mut e = g.depth_render_encoder(&cb, &s.target, &o.depth);
            g.set_render_pipeline(&mut e, &o.depth_state);
            g.set_vertex_buffer(&mut e, &s.rects[0], 0, 0);
            g.set_fragment_texture(&mut e, &s.textures[0], 0);
            g.set_fragment_sampler(&mut e, &s.sampler, 0);
            (cb, Some(e))
        },
        |state: &mut (G::CommandBuffer, Option<G::RenderEncoder>), i| {
            g.set_depth_stencil(
                state.1.as_mut().expect("the window's encoder"),
                &o.depth_stencils[i % 2],
            )
        },
        |(cb, e)| {
            g.end_render(e.expect("the window's encoder"));
            g.commit(&cb);
            g.wait(&cb);
        },
    );

    // The compute encoder's calls.
    let compute = |name: &str, calls: usize, call: &dyn Fn(&mut G::ComputeEncoder, usize)| {
        group(
            g,
            name,
            (calls, w),
            || {
                let cb = g.command_buffer();
                let mut e = g.compute_encoder(&cb);
                g.set_compute_pipeline(&mut e, &s.tiny[0]);
                g.set_buffer(&mut e, &o.output, 0, 0);
                g.set_buffer(&mut e, &o.input, 0, 1);
                (cb, Some(e))
            },
            |state: &mut (G::CommandBuffer, Option<G::ComputeEncoder>), i| {
                call(state.1.as_mut().expect("the window's encoder"), i)
            },
            |(cb, e)| {
                g.end_compute(e.expect("the window's encoder"));
                g.commit(&cb);
                g.wait(&cb);
            },
        );
    };
    compute("set_compute_pipeline", 256, &|e, i| {
        g.set_compute_pipeline(e, &s.tiny[i % 2])
    });
    compute("set_buffer", 256, &|e, i| {
        g.set_buffer(e, &o.output, (i % 64) * 256, 0)
    });
    compute("set_texture", 256, &|e, i| {
        g.set_texture(e, &s.textures[i % s.textures.len()], 0)
    });
    compute("dispatch", 256, &|e, _| g.dispatch(e, 1, 64));
    compute("use_resources_read_256", 1, &|e, _| {
        g.use_buffers_read(e, &resident)
    });
    compute("use_resources_read_1", 1, &|e, _| {
        g.use_buffers_read(e, &[&o.resident[0]])
    });
    compute("use_resource_read_write", 1, &|e, _| {
        g.use_buffer(e, &o.resident[0])
    });
    compute("use_heap", 1, &|e, _| g.use_heap(e, &o.heap));
    compute("update_fence", 1, &|e, _| g.update_fence(e, &o.fence));
    // The fence this encoder waits for was updated by the encoder before it in the same command
    // buffer, as Metal's fences are used, so nothing waits for an update that never comes.
    group(
        g,
        "wait_for_fence",
        (1, lone),
        || {
            let cb = g.command_buffer();
            let mut first = g.compute_encoder(&cb);
            g.set_compute_pipeline(&mut first, &s.tiny[0]);
            g.set_buffer(&mut first, &o.output, 0, 0);
            g.set_buffer(&mut first, &o.input, 0, 1);
            g.dispatch(&mut first, 1, 64);
            g.update_fence(&mut first, &o.fence);
            g.end_compute(first);
            let mut e = g.compute_encoder(&cb);
            g.set_compute_pipeline(&mut e, &s.tiny[0]);
            g.set_buffer(&mut e, &o.output, 0, 0);
            g.set_buffer(&mut e, &o.input, 0, 1);
            (cb, Some(e))
        },
        |state: &mut (G::CommandBuffer, Option<G::ComputeEncoder>), _| {
            g.wait_for_fence(state.1.as_mut().expect("the window's encoder"), &o.fence)
        },
        |(cb, e)| {
            let mut e = e.expect("the window's encoder");
            g.dispatch(&mut e, 1, 64);
            g.end_compute(e);
            g.commit(&cb);
            g.wait(&cb);
        },
    );

    // The blit encoder's calls, and the whole blit encoder one fill takes.
    let blit = |name: &str, calls: usize, call: &dyn Fn(&mut G::BlitEncoder, usize)| {
        group(
            g,
            name,
            (calls, w),
            || {
                let cb = g.command_buffer();
                let e = g.blit_encoder(&cb);
                (cb, Some(e))
            },
            |state: &mut (G::CommandBuffer, Option<G::BlitEncoder>), i| {
                call(state.1.as_mut().expect("the window's encoder"), i)
            },
            |(cb, e)| {
                g.end_blit(e.expect("the window's encoder"));
                g.commit(&cb);
                g.wait(&cb);
            },
        );
    };
    blit("copy_buffer_1k", 256, &|e, i| {
        let at = (i % 256) * 1024;
        g.copy_buffer(e, (&o.from, at), (&o.to, at), 1024)
    });
    blit("fill_buffer_256", 256, &|e, i| {
        g.fill_buffer_in(e, &o.to, (i % 256) * 1024, 256)
    });
    blit("copy_buffer_to_texture", 16, &|e, _| {
        g.copy_to_texture(
            e,
            &o.small,
            &o.small_texture,
            (SMALL_SIDE, SMALL_SIDE),
            SMALL_SIDE * 4,
        )
    });
    group(
        g,
        "fill_in_own_encoder",
        (16, w),
        || g.command_buffer(),
        |cb: &mut G::CommandBuffer, _| g.fill(cb, &o.to, 256),
        |cb| {
            g.commit(&cb);
            g.wait(&cb);
        },
    );

    // The objects a frame makes and drops: each window's are released before the next window's,
    // so no group runs in memory the one before it filled.
    group(
        g,
        "buffer_4k",
        (64, w),
        || Vec::with_capacity(64),
        |held: &mut Vec<G::Buffer>, _| held.push(g.buffer(4096)),
        drop,
    );
    group(
        g,
        "buffer_1m",
        (16, w),
        || Vec::with_capacity(16),
        |held: &mut Vec<G::Buffer>, _| held.push(g.buffer(1 << 20)),
        drop,
    );
    group(
        g,
        "buffer_with_bytes_4k",
        (64, w),
        || Vec::with_capacity(64),
        |held: &mut Vec<G::Buffer>, _| held.push(g.buffer_with(&o.texels[..4096])),
        drop,
    );
    group(
        g,
        "buffer_no_copy_1m",
        (NO_COPY_BUFFERS, w),
        || Vec::with_capacity(NO_COPY_BUFFERS),
        |held: &mut Vec<G::Buffer>, i| {
            // SAFETY: the pages are page-aligned read-write memory of this many bytes, and they
            // outlive every buffer made over them (`Objects::pages`).
            held.push(unsafe { g.buffer_no_copy(o.pages.add(i * NO_COPY_EACH), NO_COPY_EACH) })
        },
        drop,
    );
    group(
        g,
        "texture_256_shared",
        (32, w),
        || Vec::with_capacity(32),
        |held: &mut Vec<G::Texture>, _| {
            held.push(g.texture(Format::Bgra8, SMALL_SIDE, SMALL_SIDE, false))
        },
        drop,
    );
    group(
        g,
        "texture_1024_target",
        (8, w),
        || Vec::with_capacity(8),
        |held: &mut Vec<G::Texture>, _| held.push(g.texture(Format::Bgra8, 1024, 1024, true)),
        drop,
    );
    group(
        g,
        "heap_texture_256",
        (HEAP_TEXTURES, w),
        || Vec::with_capacity(HEAP_TEXTURES),
        |held: &mut Vec<G::Texture>, i| {
            held.push(g.heap_texture(&o.heap, SMALL_SIDE, SMALL_SIDE, i * SMALL_BYTES))
        },
        drop,
    );
    group(
        g,
        "buffer_texture_256",
        (16, w),
        || Vec::with_capacity(16),
        |held: &mut Vec<G::Texture>, _| {
            held.push(g.buffer_texture(&o.small, (SMALL_SIDE, SMALL_SIDE), 0, SMALL_SIDE * 4))
        },
        drop,
    );
    group(
        g,
        "sampler",
        (32, w),
        || Vec::with_capacity(32),
        |held: &mut Vec<G::Sampler>, i| held.push(g.sampler(i % 2 == 0)),
        drop,
    );
    group(
        g,
        "shared_event",
        (16, w),
        || Vec::with_capacity(16),
        |held: &mut Vec<G::SharedEvent>, _| held.push(g.shared_event()),
        drop,
    );
    group(
        g,
        "fence",
        (16, w),
        || Vec::with_capacity(16),
        |held: &mut Vec<G::Fence>, _| held.push(g.fence()),
        drop,
    );
    group(
        g,
        "event_value",
        (256, w),
        || (),
        |_, _| {
            std::hint::black_box(g.event_value(&o.event));
        },
        drop,
    );
    let next_value = std::cell::Cell::new(1u64);
    group(
        g,
        "set_event_value",
        (256, w),
        || (),
        |_, _| {
            let value = next_value.get();
            next_value.set(value + 1);
            g.set_event_value(&o.event, value)
        },
        drop,
    );

    // The bytes a frame reads back and writes in, with no GPU work writing them: Metal's
    // `getBytes` and `contents` wait for nothing, and nor may ours.
    group(
        g,
        "read_buffer_contents_256",
        (256, w),
        || vec![0u8; 256],
        |bytes: &mut Vec<u8>, _| g.read_buffer(&o.input, bytes),
        drop,
    );
    group(
        g,
        "get_bytes_256",
        (4, w),
        || vec![0u8; SMALL_BYTES],
        |bytes: &mut Vec<u8>, _| {
            g.get_bytes(
                &o.small_texture,
                (SMALL_SIDE, SMALL_SIDE),
                bytes,
                SMALL_SIDE * 4,
            )
        },
        drop,
    );
    group(
        g,
        "replace_region_256",
        (4, w),
        || (),
        |_, _| {
            g.replace_region(
                &o.small_texture,
                (SMALL_SIDE, SMALL_SIDE),
                &o.texels,
                SMALL_SIDE * 4,
            )
        },
        drop,
    );

    // The driver's own cost under the calls above, where the side has one (`vkfloor.rs`).
    // SAFETY: the pages are this process's, eight megabytes of them, and they outlive the call.
    unsafe { g.driver_floor(w, o.pages) };

    // Nothing above may have made a pipeline: the groups bind states made before them, and a
    // making is pipeline-creation time, which has its own ceiling and no place in a call's cost.
    let made_after = g.work_counts();
    for (counted, before) in &made_before {
        if *counted != "variant_pipelines_made" {
            continue;
        }
        let now = made_after
            .iter()
            .find(|(name, _)| name == counted)
            .map_or(0.0, |(_, count)| *count);
        emit(
            "percall",
            "meta_pipelines_made",
            "count",
            &mut [now - before],
        );
    }
}
