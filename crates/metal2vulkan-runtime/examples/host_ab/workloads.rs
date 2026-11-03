//! The workloads, written once against [`Gpu`]. Each prints its measurements as result lines
//! ([`emit`]): `R`, the workload, the metric, the unit, the median, the 10th and 90th percentiles
//! and the sample count, tab-separated.

use crate::gpu::{Format, Gpu};
use std::path::Path;
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// How hard the workloads run.
#[derive(Clone, Copy, Debug)]
pub struct Sizes {
    /// Draws in a render pass, and dispatches in a compute encoder, of the encode workload.
    pub draws: usize,
    /// Timed command buffers of the encode workload, and of each phase of `fixed`.
    pub timed: usize,
    /// Command buffers of each latency phase.
    pub latency: usize,
    /// Loop iterations of the heavy shaders.
    pub heavy_iterations: u32,
    /// Timed command buffers of each heavy shader.
    pub heavy_runs: usize,
    /// Draws in the many-draws workload's frame.
    pub many_draws: usize,
    /// Timed frames of the compositing workload.
    pub frames: usize,
}

/// One result line (module docs) of `values`, which it sorts.
pub fn emit(workload: &str, metric: &str, unit: &str, values: &mut [f64]) {
    assert!(!values.is_empty(), "{workload} {metric}: no samples");
    values.sort_by(f64::total_cmp);
    let at = |p: f64| values[((values.len() - 1) as f64 * p).round() as usize];
    println!(
        "R\t{workload}\t{metric}\t{unit}\t{:.3}\t{:.3}\t{:.3}\t{}",
        at(0.5),
        at(0.1),
        at(0.9),
        values.len()
    );
}

fn us(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// The process's CPU time so far (user and system, every thread).
pub fn process_cpu() -> Duration {
    // SAFETY: `getrusage` writes the struct it is given.
    let usage = unsafe {
        let mut usage = std::mem::zeroed::<libc::rusage>();
        assert_eq!(libc::getrusage(libc::RUSAGE_SELF, &mut usage), 0);
        usage
    };
    let tv = |t: libc::timeval| Duration::new(t.tv_sec as u64, t.tv_usec as u32 * 1000);
    tv(usage.ru_utime) + tv(usage.ru_stime)
}

/// The calling thread's CPU time so far.
pub fn thread_cpu() -> Duration {
    // SAFETY: `clock_gettime` writes the struct it is given.
    let spec = unsafe {
        let mut spec = std::mem::zeroed::<libc::timespec>();
        assert_eq!(
            libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut spec),
            0
        );
        spec
    };
    Duration::new(spec.tv_sec as u64, spec.tv_nsec as u32)
}

/// The rectangles' buffers: [`RECTS`] rectangles each, one every [`RECT_STRIDE`] bytes.
const RECT_BUFFERS: usize = 4;
const RECTS: usize = 1024;
const RECT_STRIDE: usize = 256;
/// The small textures the draws take in turn, and their side.
const TEXTURES: usize = 16;
const TEXTURE_SIDE: usize = 64;
/// The small draws' target's side.
pub const TARGET_SIDE: usize = 512;
/// The heavy fragment's target's side, its texture's side, and the heavy kernel's texture's.
const HEAVY_TARGET_SIDE: usize = 2048;
const HEAVY_TEXTURE_SIDE: usize = 1024;
const KERNEL_TEXTURE_SIDE: usize = 256;
/// The heavy kernel's grid: threadgroups of 256 threads.
const HEAVY_GROUPS: usize = 4096;
/// The feature kernels (`host_ab.metal`).
const FEATURES: [&str; 9] = [
    "k_fma",
    "k_sin",
    "k_half",
    "k_constant",
    "k_device",
    "k_texture",
    "k_texture_float",
    "k_threadgroup_float",
    "k_threadgroup_half",
];
const HEAVY_THREADS: usize = 256;
/// The heavy and feature kernels' output: one `float4` per thread.
const HEAVY_OUT_BYTES: usize = HEAVY_GROUPS * HEAVY_THREADS * 16;

/// What the workloads draw and dispatch with, made once per process.
pub struct Scene<G: Gpu> {
    /// `quad_vertex` + `textured_fragment` into `BGRA8Unorm`, without and with blending.
    pub textured: [G::RenderPipeline; 2],
    /// `quad_vertex` + `heavy_fragment` into `RGBA16Float`.
    pub heavy_fragment: G::RenderPipeline,
    pub heavy_kernel: G::ComputePipeline,
    /// The feature kernels (`host_ab.metal`), by name.
    pub features: Vec<(&'static str, G::ComputePipeline)>,
    /// `tiny_kernel` and `tiny_kernel_b`.
    pub tiny: [G::ComputePipeline; 2],
    pub textures: Vec<G::Texture>,
    pub heavy_texture: G::Texture,
    pub kernel_texture: G::Texture,
    pub sampler: G::Sampler,
    pub target: G::Texture,
    pub heavy_target: G::Texture,
    pub rects: Vec<G::Buffer>,
    /// One rectangle over the whole target.
    pub full: G::Buffer,
    /// The heavy fragment's parameters, and the heavy kernel's (16 times the iterations).
    pub params: G::Buffer,
    pub kernel_params: G::Buffer,
    pub tiny_out: G::Buffer,
    pub tiny_values: G::Buffer,
    pub heavy_out: G::Buffer,
    /// The feature kernels' `src`.
    pub heavy_in: G::Buffer,
    pub fill: G::Buffer,
}

/// A 32 by 32 grid of small rectangles over the target, each at its own 256-byte offset (a
/// constant buffer's offset alignment), in the order of rectangle buffer `b`.
fn rect_bytes(b: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; RECTS * RECT_STRIDE];
    for r in 0..RECTS {
        let cell = (r + b * 7) % RECTS;
        let (x, y) = ((cell % 32) as f32, (cell / 32) as f32);
        let at = |v: f32| v / 16.0 - 1.0;
        let rect = floats(&[at(x), at(y), at(x + 0.5), at(y + 0.5)]);
        bytes[r * RECT_STRIDE..r * RECT_STRIDE + 16].copy_from_slice(&rect);
    }
    bytes
}

fn texels(side: usize, seed: u8) -> Vec<u8> {
    (0..side * side)
        .flat_map(|i| {
            let (x, y) = ((i % side) as u8, (i / side) as u8);
            [x.wrapping_mul(3) ^ seed, y.wrapping_mul(5), x ^ y, 255]
        })
        .collect()
}

fn floats(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The heavy shaders' `HeavyParams`: 64 `float4`, the iterations and a scale.
pub fn params_bytes(iterations: u32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(1040);
    for i in 0..64 {
        let f = i as f32;
        bytes.extend(floats(&[
            0.97 - f * 0.001,
            1.3 + f * 0.01,
            0.25 + f * 0.002,
            0.003 * f,
        ]));
    }
    bytes.extend(iterations.to_le_bytes());
    bytes.extend(0.5f32.to_le_bytes());
    bytes.resize(1040, 0);
    bytes
}

impl<G: Gpu> Scene<G> {
    pub fn new(g: &G, library: &G::Library, sizes: &Sizes) -> Scene<G> {
        let textured = [
            g.render_pipeline(
                library,
                "quad_vertex",
                "textured_fragment",
                Format::Bgra8,
                false,
            ),
            g.render_pipeline(
                library,
                "quad_vertex",
                "textured_fragment",
                Format::Bgra8,
                true,
            ),
        ];
        let heavy_fragment = g.render_pipeline(
            library,
            "quad_vertex",
            "heavy_fragment",
            Format::Rgba16Float,
            false,
        );
        let heavy_kernel = g.compute_pipeline(library, "heavy_kernel");
        let features = FEATURES
            .iter()
            .map(|&name| (name, g.compute_pipeline(library, name)))
            .collect();
        let tiny = [
            g.compute_pipeline(library, "tiny_kernel"),
            g.compute_pipeline(library, "tiny_kernel_b"),
        ];
        let shader_texture = |side: usize, seed: u8| {
            let texture = g.texture(Format::Rgba8, side, side, false);
            g.replace_region(&texture, (side, side), &texels(side, seed), side * 4);
            texture
        };
        let textures = (0..TEXTURES)
            .map(|n| shader_texture(TEXTURE_SIDE, n as u8 * 16))
            .collect();
        let rects = (0..RECT_BUFFERS)
            .map(|b| g.buffer_with(&rect_bytes(b)))
            .collect();
        Scene {
            textured,
            heavy_fragment,
            heavy_kernel,
            features,
            tiny,
            textures,
            heavy_texture: shader_texture(HEAVY_TEXTURE_SIDE, 7),
            kernel_texture: shader_texture(KERNEL_TEXTURE_SIDE, 9),
            sampler: g.sampler(true),
            target: g.texture(Format::Bgra8, TARGET_SIDE, TARGET_SIDE, true),
            heavy_target: g.texture(
                Format::Rgba16Float,
                HEAVY_TARGET_SIDE,
                HEAVY_TARGET_SIDE,
                true,
            ),
            rects,
            full: g.buffer_with(&floats(&[-1.0, -1.0, 1.0, 1.0])),
            params: g.buffer_with(&params_bytes(sizes.heavy_iterations)),
            kernel_params: g.buffer_with(&params_bytes(sizes.heavy_iterations * 16)),
            tiny_out: g.buffer(RECTS * RECT_STRIDE),
            tiny_values: g.buffer_with(&vec![1u8; RECTS * RECT_STRIDE]),
            heavy_out: g.buffer(HEAVY_OUT_BYTES),
            heavy_in: g.buffer_with(&vec![0x3c; HEAVY_OUT_BYTES]),
            fill: g.buffer(4096),
        }
    }
}

/// What the encode workload changes between draws (and dispatches).
#[derive(Clone, Copy)]
enum Churn {
    /// Nothing: every draw the same.
    Static,
    /// The vertex buffer's offset alone (`setVertexBufferOffset:atIndex:`).
    Offset,
    /// The vertex buffer (another buffer and offset) and the fragment texture.
    Bind,
    /// As `Bind`, and the pipeline state switched between two.
    Pipeline,
    /// As `Pipeline`, in render passes of [`PASS_DRAWS`] draws each (the pass loads the target,
    /// the first clears it): a window server's few draws a pass, each binding its own.
    Passes,
}

/// The draws of each pass of [`Churn::Passes`].
const PASS_DRAWS: usize = 3;

impl Churn {
    fn name(self) -> &'static str {
        match self {
            Churn::Static => "static",
            Churn::Offset => "offset",
            Churn::Bind => "bind",
            Churn::Pipeline => "pipeline",
            Churn::Passes => "passes",
        }
    }
}

/// Encode one render pass of `draws` small draws with `churn` (module docs of `main.rs`).
fn encode_draws<G: Gpu>(
    g: &G,
    s: &Scene<G>,
    command_buffer: &G::CommandBuffer,
    draws: usize,
    churn: Churn,
) {
    let mut encoder = g.render_encoder(command_buffer, &s.target, true);
    g.set_render_pipeline(&mut encoder, &s.textured[0]);
    g.set_fragment_sampler(&mut encoder, &s.sampler, 0);
    g.set_vertex_buffer(&mut encoder, &s.rects[0], 0, 0);
    g.set_fragment_texture(&mut encoder, &s.textures[0], 0);
    for draw in 0..draws {
        if matches!(churn, Churn::Passes) && draw > 0 && draw % PASS_DRAWS == 0 {
            g.end_render(encoder);
            encoder = g.render_encoder(command_buffer, &s.target, false);
            g.set_fragment_sampler(&mut encoder, &s.sampler, 0);
        }
        if matches!(churn, Churn::Offset) {
            g.set_vertex_buffer_offset(&mut encoder, (draw % RECTS) * RECT_STRIDE, 0);
        }
        if !matches!(churn, Churn::Static | Churn::Offset) {
            let buffer = &s.rects[(draw / 7) % RECT_BUFFERS];
            g.set_vertex_buffer(&mut encoder, buffer, (draw % RECTS) * RECT_STRIDE, 0);
            g.set_fragment_texture(&mut encoder, &s.textures[draw % TEXTURES], 0);
        }
        if matches!(churn, Churn::Pipeline | Churn::Passes) {
            g.set_render_pipeline(&mut encoder, &s.textured[draw % 2]);
        }
        g.draw_strip(&mut encoder, 4);
    }
    g.end_render(encoder);
}

/// Encode one compute encoder of `dispatches` small dispatches, changing the buffers' offsets and
/// the pipeline state at each where `churn`.
fn encode_dispatches<G: Gpu>(
    g: &G,
    s: &Scene<G>,
    command_buffer: &G::CommandBuffer,
    dispatches: usize,
    churn: DispatchChurn,
) {
    let mut encoder = g.compute_encoder(command_buffer);
    g.set_compute_pipeline(&mut encoder, &s.tiny[0]);
    g.set_buffer(&mut encoder, &s.tiny_out, 0, 0);
    g.set_buffer(&mut encoder, &s.tiny_values, 0, 1);
    for dispatch in 0..dispatches {
        if churn == DispatchChurn::Churn {
            g.set_compute_pipeline(&mut encoder, &s.tiny[dispatch % 2]);
        }
        if churn != DispatchChurn::Static {
            let offset = (dispatch % RECTS) * RECT_STRIDE;
            g.set_buffer(&mut encoder, &s.tiny_out, offset, 0);
            g.set_buffer(&mut encoder, &s.tiny_values, offset, 1);
        }
        g.dispatch(&mut encoder, 1, 64);
    }
    g.end_compute(encoder);
}

/// What the encode workload's dispatches change between them.
#[derive(Clone, Copy, PartialEq, Eq)]
enum DispatchChurn {
    /// Nothing.
    Static,
    /// Both buffers' offsets.
    Offsets,
    /// The pipeline state and both buffers' offsets.
    Churn,
}

impl DispatchChurn {
    fn name(self) -> &'static str {
        match self {
            DispatchChurn::Static => "static",
            DispatchChurn::Offsets => "offsets",
            DispatchChurn::Churn => "churn",
        }
    }
}

/// Encode the encode workload's draws with `churn` on `threads` threads at once, each through its
/// own queue (`Gpu::with_own_queue`) into its own target, of its own rectangles, texture and
/// sampler, through `shared` where given, else through a pipeline state of its own made like it, `rounds` command buffers of `draws` draws
/// each, one in flight while the next encodes: the draws a second of all of them together, from
/// their common start to the last one's last commit, the mean of each thread's CPU time per draw,
/// and the process's CPU time per draw (module docs of `main.rs`).
fn threaded_draws<G: Gpu>(
    g: &G,
    library: &G::Library,
    (threads, shared): (usize, Option<&G::RenderPipeline>),
    (rounds, draws, churn): (usize, usize, Churn),
) -> (f64, f64, f64) {
    let start = std::sync::Barrier::new(threads + 1);
    let (per_thread, wall, cpu) = std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                let start = &start;
                scope.spawn(move || {
                    let g = g.with_own_queue();
                    let target = g.texture(Format::Bgra8, 256, 256, true);
                    let rects = g.buffer_with(&rect_bytes(0));
                    let texture = g.texture(Format::Rgba8, TEXTURE_SIDE, TEXTURE_SIDE, false);
                    let side = (TEXTURE_SIDE, TEXTURE_SIDE);
                    g.replace_region(&texture, side, &texels(TEXTURE_SIDE, 0), TEXTURE_SIDE * 4);
                    let sampler = g.sampler(true);
                    let own = shared.is_none().then(|| {
                        let (vertex, fragment) = ("quad_vertex", "textured_fragment");
                        g.render_pipeline(library, vertex, fragment, Format::Bgra8, false)
                    });
                    let state = shared.or(own.as_ref()).expect("a pipeline state");
                    let encode = |command_buffer: &G::CommandBuffer| {
                        let mut encoder = g.render_encoder(command_buffer, &target, true);
                        g.set_render_pipeline(&mut encoder, state);
                        g.set_fragment_sampler(&mut encoder, &sampler, 0);
                        g.set_vertex_buffer(&mut encoder, &rects, 0, 0);
                        g.set_fragment_texture(&mut encoder, &texture, 0);
                        for draw in 0..draws {
                            if matches!(churn, Churn::Offset) {
                                let offset = (draw % RECTS) * RECT_STRIDE;
                                g.set_vertex_buffer_offset(&mut encoder, offset, 0);
                            }
                            g.draw_strip(&mut encoder, 4);
                        }
                        g.end_render(encoder);
                    };
                    // Warm: its pipeline made, its queue's first command buffers behind it.
                    for _ in 0..3 {
                        g.pool(|| {
                            let command_buffer = g.command_buffer();
                            encode(&command_buffer);
                            g.commit(&command_buffer);
                            g.wait(&command_buffer);
                        });
                    }
                    start.wait();
                    let cpu0 = thread_cpu();
                    let mut in_flight: Option<G::CommandBuffer> = None;
                    for _ in 0..rounds {
                        g.pool(|| {
                            let command_buffer = g.command_buffer();
                            encode(&command_buffer);
                            if let Some(last) = in_flight.take() {
                                g.wait(&last);
                            }
                            g.commit(&command_buffer);
                            in_flight = Some(command_buffer);
                        });
                    }
                    let (done, cpu) = (Instant::now(), thread_cpu() - cpu0);
                    if let Some(last) = in_flight.take() {
                        g.wait(&last);
                    }
                    (done, cpu)
                })
            })
            .collect();
        start.wait();
        let (t0, cpu0) = (Instant::now(), process_cpu());
        let ended: Vec<(Instant, Duration)> = workers
            .into_iter()
            .map(|worker| worker.join().expect("a drawing thread"))
            .collect();
        let cpu = process_cpu() - cpu0;
        let last = ended.iter().map(|(done, _)| *done).max().expect("a thread");
        let per_thread: Duration = ended.iter().map(|(_, cpu)| *cpu).sum();
        (per_thread / threads as u32, last - t0, cpu)
    });
    let each = (rounds * draws) as f64;
    let all = each * threads as f64;
    (
        all / wall.as_secs_f64(),
        us(per_thread) / each,
        us(cpu) / all,
    )
}

/// The thread-scaling workload, its shared pipeline state `shared` (module docs of `main.rs`).
pub fn threads<G: Gpu>(g: &G, library: &G::Library, shared: &G::RenderPipeline, sizes: &Sizes) {
    let rounds = (sizes.timed / 4).max(8);
    for churn in [Churn::Static, Churn::Offset] {
        for (pso, shared) in [("shared", Some(shared)), ("each", None)] {
            for threads in [1, 2, 4, 8] {
                let (mut rate, mut thread, mut process) = (vec![], vec![], vec![]);
                for _ in 0..3 {
                    let run = (rounds, sizes.draws, churn);
                    let (r, t, p) = threaded_draws(g, library, (threads, shared), run);
                    rate.push(r / 1e6);
                    thread.push(t);
                    process.push(p);
                }
                let name = format!("draw_{}_{pso}_{threads}t", churn.name());
                emit("threads", &format!("{name}_rate"), "Mdraws/s", &mut rate);
                emit(
                    "threads",
                    &format!("{name}_thread_cpu"),
                    "us/draw",
                    &mut thread,
                );
                emit(
                    "threads",
                    &format!("{name}_process_cpu"),
                    "us/draw",
                    &mut process,
                );
            }
        }
    }
}

/// The CPU cost of encoding: per draw and per dispatch, on the encoding thread (wall) and in the
/// whole process (CPU time of every thread, from the command buffer's creation until its wait
/// returned, which takes in work moved off the encoding thread to commit or to other threads).
pub fn encode<G: Gpu>(
    g: &G,
    s: &Scene<G>,
    library: &G::Library,
    sizes: &Sizes,
    variants: &[String],
) {
    let n = sizes.draws;
    let runs = |name: String| variants.is_empty() || variants.contains(&name);
    for churn in [
        Churn::Static,
        Churn::Offset,
        Churn::Bind,
        Churn::Pipeline,
        Churn::Passes,
    ] {
        if !runs(format!("draw_{}", churn.name())) {
            continue;
        }
        let (mut wall, mut cpu, mut commit, mut total) = (vec![], vec![], vec![], vec![]);
        for run in 0..sizes.timed + 3 {
            g.pool(|| {
                let cpu0 = process_cpu();
                let t0 = Instant::now();
                let command_buffer = g.command_buffer();
                encode_draws(g, s, &command_buffer, n, churn);
                let t1 = Instant::now();
                g.commit(&command_buffer);
                let t2 = Instant::now();
                g.wait(&command_buffer);
                let t3 = Instant::now();
                let cpu1 = process_cpu();
                if run >= 3 {
                    wall.push(us(t1 - t0) / n as f64);
                    cpu.push(us(cpu1 - cpu0) / n as f64);
                    commit.push(us(t2 - t1));
                    total.push(ms(t3 - t0));
                }
            });
        }
        let name = churn.name();
        emit(
            "encode",
            &format!("draw_{name}_encode"),
            "us/draw",
            &mut wall,
        );
        emit(
            "encode",
            &format!("draw_{name}_process_cpu"),
            "us/draw",
            &mut cpu,
        );
        emit(
            "encode",
            &format!("draw_{name}_commit_call"),
            "us",
            &mut commit,
        );
        emit("encode", &format!("draw_{name}_cb_total"), "ms", &mut total);
    }
    for churn in [
        DispatchChurn::Static,
        DispatchChurn::Offsets,
        DispatchChurn::Churn,
    ] {
        if !runs(format!("dispatch_{}", churn.name())) {
            continue;
        }
        let (mut wall, mut cpu, mut total) = (vec![], vec![], vec![]);
        let mut counts = None;
        for run in 0..sizes.timed + 3 {
            if run == 3 {
                counts = Some(g.work_counts());
            }
            g.pool(|| {
                let cpu0 = process_cpu();
                let t0 = Instant::now();
                let command_buffer = g.command_buffer();
                encode_dispatches(g, s, &command_buffer, n, churn);
                let t1 = Instant::now();
                g.commit(&command_buffer);
                g.wait(&command_buffer);
                let t3 = Instant::now();
                let cpu1 = process_cpu();
                if run >= 3 {
                    wall.push(us(t1 - t0) / n as f64);
                    cpu.push(us(cpu1 - cpu0) / n as f64);
                    total.push(ms(t3 - t0));
                }
            });
        }
        let name = churn.name();
        emit(
            "encode",
            &format!("dispatch_{name}_encode"),
            "us/dispatch",
            &mut wall,
        );
        emit(
            "encode",
            &format!("dispatch_{name}_process_cpu"),
            "us/dispatch",
            &mut cpu,
        );
        emit(
            "encode",
            &format!("dispatch_{name}_cb_total"),
            "ms",
            &mut total,
        );
        let dispatches = (sizes.timed * n) as f64;
        emit_memo_shares(g, counts, dispatches, &format!("dispatch_{name}"));
    }
    encode_resident(g, s, sizes, &runs);
    encode_handles(g, s, library, sizes, &runs);
}

/// The encode workload's `dispatch_handles` variant: a compute encoder of `draws`
/// single-threadgroup dispatches of `tiny_handle_kernel`, which reads a texture made resident
/// through its handle in an argument buffer, so each dispatch binds the device's texture table.
/// Per dispatch: the encoding thread's wall time and the process's CPU time.
fn encode_handles<G: Gpu>(
    g: &G,
    s: &Scene<G>,
    library: &G::Library,
    sizes: &Sizes,
    runs: &dyn Fn(String) -> bool,
) {
    if !runs("dispatch_handles".to_owned()) {
        return;
    }
    let n = sizes.draws;
    let pipeline = g.compute_pipeline(library, "tiny_handle_kernel");
    let texture = &s.textures[0];
    let handles = g.buffer_with(&g.texture_id(texture).to_le_bytes());
    let (mut wall, mut cpu) = (vec![], vec![]);
    for run in 0..sizes.timed + 3 {
        g.pool(|| {
            let cpu0 = process_cpu();
            let command_buffer = g.command_buffer();
            let mut encoder = g.compute_encoder(&command_buffer);
            g.set_compute_pipeline(&mut encoder, &pipeline);
            g.set_buffer(&mut encoder, &s.tiny_out, 0, 0);
            g.set_buffer(&mut encoder, &handles, 0, 1);
            g.use_textures(&mut encoder, &[texture]);
            let t0 = Instant::now();
            for _ in 0..n {
                g.dispatch(&mut encoder, 1, 64);
            }
            let t1 = Instant::now();
            g.end_compute(encoder);
            g.commit(&command_buffer);
            g.wait(&command_buffer);
            let cpu1 = process_cpu();
            if run >= 3 {
                wall.push(us(t1 - t0) / n as f64);
                cpu.push(us(cpu1 - cpu0) / n as f64);
            }
        });
    }
    emit(
        "encode",
        "dispatch_handles_encode",
        "us/dispatch",
        &mut wall,
    );
    emit(
        "encode",
        "dispatch_handles_process_cpu",
        "us/dispatch",
        &mut cpu,
    );
}

/// The resident buffers of the encode workload's `dispatch_resident` variants, of 256 bytes each.
const RESIDENT: usize = 4096;

/// The encode workload's `dispatch_resident_*` variants: a compute encoder whose pipeline state
/// and buffers are set, one `useResources:count:usage:` (reading) of none, 1, 256 or 4096
/// buffers, or of one buffer 4096 times (`same_4096`), then `draws` single-threadgroup dispatches
/// of the tiny kernel; `moving_0`, `_16`, `_256` and `_4096` as many after none, 16, 256 or 4096,
/// each dispatch after the output buffer's offset moved; `textures_256` and `_4096` as many after
/// 256 or 4096 4-by-4 textures made resident. Per variant: the `useResources` call, the encoding
/// thread's wall time per dispatch, the process's CPU time per dispatch, and the command buffer's
/// whole time.
fn encode_resident<G: Gpu>(g: &G, s: &Scene<G>, sizes: &Sizes, runs: &dyn Fn(String) -> bool) {
    // (name, resident buffers or textures, one buffer `count` times, the output's offset moving,
    // textures).
    let variants = [
        ("0", 0, false, false, false),
        ("1", 1, false, false, false),
        ("256", 256, false, false, false),
        ("4096", 4096, false, false, false),
        ("same_4096", 4096, true, false, false),
        ("moving_0", 0, false, true, false),
        ("moving_16", 16, false, true, false),
        ("moving_256", 256, false, true, false),
        ("moving_4096", 4096, false, true, false),
        ("textures_256", 256, false, false, true),
        ("textures_4096", 4096, false, false, true),
    ];
    let wanted: Vec<_> = (variants.iter())
        .filter(|(name, ..)| runs(format!("dispatch_resident_{name}")))
        .collect();
    if wanted.is_empty() {
        return;
    }
    let n = sizes.draws;
    let resident: Vec<G::Buffer> = (0..RESIDENT).map(|_| g.buffer(256)).collect();
    let textures: Vec<G::Texture> = match wanted.iter().any(|(.., textures)| *textures) {
        true => (0..RESIDENT)
            .map(|_| g.texture(Format::Rgba8, 4, 4, false))
            .collect(),
        false => Vec::new(),
    };
    for &&(name, count, same, moving, of_textures) in &wanted {
        let list: Vec<&G::Buffer> = match (same, of_textures) {
            (_, true) => Vec::new(),
            (true, false) => vec![&resident[0]; count],
            (false, false) => resident[..count].iter().collect(),
        };
        let texture_list: Vec<&G::Texture> = match of_textures {
            true => textures[..count].iter().collect(),
            false => Vec::new(),
        };
        let (mut using, mut wall, mut cpu, mut total) = (vec![], vec![], vec![], vec![]);
        let mut counts = None;
        for run in 0..sizes.timed + 3 {
            if run == 3 {
                counts = Some(g.work_counts());
            }
            g.pool(|| {
                let cpu0 = process_cpu();
                let t0 = Instant::now();
                let command_buffer = g.command_buffer();
                let mut encoder = g.compute_encoder(&command_buffer);
                g.set_compute_pipeline(&mut encoder, &s.tiny[0]);
                g.set_buffer(&mut encoder, &s.tiny_out, 0, 0);
                g.set_buffer(&mut encoder, &s.tiny_values, 0, 1);
                let t1 = Instant::now();
                match of_textures {
                    true => g.use_textures(&mut encoder, &texture_list),
                    false => g.use_buffers(&mut encoder, &list),
                }
                let t2 = Instant::now();
                for dispatch in 0..n {
                    if moving {
                        let offset = (dispatch % RECTS) * RECT_STRIDE;
                        g.set_buffer(&mut encoder, &s.tiny_out, offset, 0);
                    }
                    g.dispatch(&mut encoder, 1, 64);
                }
                let t3 = Instant::now();
                g.end_compute(encoder);
                g.commit(&command_buffer);
                g.wait(&command_buffer);
                let t4 = Instant::now();
                let cpu1 = process_cpu();
                if run >= 3 {
                    using.push(us(t2 - t1));
                    wall.push(us(t3 - t2) / n as f64);
                    cpu.push(us(cpu1 - cpu0) / n as f64);
                    total.push(ms(t4 - t0));
                }
            });
        }
        let metric = |what: &str| format!("dispatch_resident_{name}_{what}");
        emit("encode", &metric("use_call"), "us", &mut using);
        emit("encode", &metric("encode"), "us/dispatch", &mut wall);
        emit("encode", &metric("process_cpu"), "us/dispatch", &mut cpu);
        emit("encode", &metric("cb_total"), "ms", &mut total);
        let dispatches = (sizes.timed * n) as f64;
        emit_memo_shares(g, counts, dispatches, &format!("dispatch_resident_{name}"));
    }
}

/// The share of `dispatches` since `counts` the runtime took as settled before
/// (`dispatches_settled_as_before`) and as planned before (`dispatches_planned_as_before`), as
/// the encode workload's `<variant>_rt_*` lines.
fn emit_memo_shares<G: Gpu>(
    g: &G,
    counts: Option<Vec<(&'static str, f64)>>,
    dispatches: f64,
    variant: &str,
) {
    let after = g.work_counts();
    for ((counted, before), (_, now)) in counts.unwrap_or_default().iter().zip(&after) {
        let metric = match *counted {
            "dispatches_settled_as_before" => "rt_settled_as_before",
            "dispatches_planned_as_before" => "rt_planned_as_before",
            _ => continue,
        };
        let share = (now - before) / dispatches;
        emit(
            "encode",
            &format!("{variant}_{metric}"),
            "share",
            &mut [share],
        );
    }
}

/// The fixed costs: a command buffer's creation, an encoder of one command (opened, bound, one
/// command, ended), and the commit call.
pub fn fixed<G: Gpu>(g: &G, s: &Scene<G>, sizes: &Sizes) {
    const ENCODERS: usize = 10;
    let (mut create, mut render, mut compute, mut blit, mut commit) =
        (vec![], vec![], vec![], vec![], vec![]);
    for run in 0..sizes.timed + 3 {
        g.pool(|| {
            let t0 = Instant::now();
            let command_buffer = g.command_buffer();
            let t1 = Instant::now();
            let mut times = [vec![], vec![], vec![]];
            for _ in 0..ENCODERS {
                let a = Instant::now();
                let mut encoder = g.render_encoder(&command_buffer, &s.target, false);
                g.set_render_pipeline(&mut encoder, &s.textured[0]);
                g.set_fragment_sampler(&mut encoder, &s.sampler, 0);
                g.set_vertex_buffer(&mut encoder, &s.rects[0], 0, 0);
                g.set_fragment_texture(&mut encoder, &s.textures[0], 0);
                g.draw_strip(&mut encoder, 4);
                g.end_render(encoder);
                let b = Instant::now();
                let mut encoder = g.compute_encoder(&command_buffer);
                g.set_compute_pipeline(&mut encoder, &s.tiny[0]);
                g.set_buffer(&mut encoder, &s.tiny_out, 0, 0);
                g.set_buffer(&mut encoder, &s.tiny_values, 0, 1);
                g.dispatch(&mut encoder, 1, 64);
                g.end_compute(encoder);
                let c = Instant::now();
                g.fill(&command_buffer, &s.fill, 256);
                let d = Instant::now();
                times[0].push(us(b - a));
                times[1].push(us(c - b));
                times[2].push(us(d - c));
            }
            let t2 = Instant::now();
            g.commit(&command_buffer);
            let t3 = Instant::now();
            g.wait(&command_buffer);
            let _ = t2;
            if run >= 3 {
                create.push(us(t1 - t0));
                render.extend(&times[0]);
                compute.extend(&times[1]);
                blit.extend(&times[2]);
                commit.push(us(t3 - t2));
            }
        });
    }
    emit("fixed", "command_buffer_create", "us", &mut create);
    emit("fixed", "render_encoder_1draw", "us", &mut render);
    emit("fixed", "compute_encoder_1dispatch", "us", &mut compute);
    emit("fixed", "blit_encoder_1fill", "us", &mut blit);
    emit("fixed", "commit_call_30_encoders", "us", &mut commit);
}

/// What a latency command buffer holds.
#[derive(Clone, Copy)]
enum Tiny {
    Empty,
    Fill,
    Dispatch,
    Draw,
}

fn encode_tiny<G: Gpu>(g: &G, s: &Scene<G>, command_buffer: &G::CommandBuffer, tiny: Tiny) {
    match tiny {
        Tiny::Empty => {}
        Tiny::Fill => g.fill(command_buffer, &s.fill, 256),
        Tiny::Dispatch => encode_dispatches(g, s, command_buffer, 1, DispatchChurn::Static),
        Tiny::Draw => encode_draws(g, s, command_buffer, 1, Churn::Static),
    }
}

/// Submit-to-completion latency of a tiny command buffer, one in flight: from the commit call
/// until `waitUntilCompleted` returns, and until a completed handler runs.
pub fn latency<G: Gpu>(g: &G, s: &Scene<G>, sizes: &Sizes) {
    for (name, tiny) in [
        ("empty", Tiny::Empty),
        ("fill", Tiny::Fill),
        ("dispatch", Tiny::Dispatch),
        ("draw", Tiny::Draw),
    ] {
        let (mut wait, mut commit) = (vec![], vec![]);
        // The host-clock timeline: commit to kernel start, kernel start to end, kernel end to
        // GPU start, GPU start to end, GPU end to the wait's return.
        let mut split: [Vec<f64>; 5] = Default::default();
        for run in 0..sizes.latency + 10 {
            g.pool(|| {
                let command_buffer = g.command_buffer();
                encode_tiny(g, s, &command_buffer, tiny);
                let h0 = crate::host_seconds();
                let t0 = Instant::now();
                g.commit(&command_buffer);
                let t1 = Instant::now();
                g.wait(&command_buffer);
                let t2 = Instant::now();
                let h2 = crate::host_seconds();
                if run >= 10 {
                    commit.push(us(t1 - t0));
                    wait.push(us(t2 - t0));
                    let [ks, ke, gs, ge] = g.times(&command_buffer);
                    let points = [h0, ks, ke, gs, ge, h2];
                    for (k, part) in split.iter_mut().enumerate() {
                        part.push((points[k + 1] - points[k]) * 1e6);
                    }
                }
            });
        }
        for (part, values) in [
            "commit_to_kernel_start",
            "kernel_start_to_end",
            "kernel_end_to_gpu_start",
            "gpu_start_to_end",
            "gpu_end_to_wait_return",
        ]
        .iter()
        .zip(&mut split)
        {
            emit("latency_split", &format!("{name}_{part}"), "us", values);
        }
        emit(
            "latency",
            &format!("{name}_commit_to_wait_return"),
            "us",
            &mut wait,
        );
        emit("latency", &format!("{name}_commit_call"), "us", &mut commit);
        let mut handler = vec![];
        for run in 0..sizes.latency + 10 {
            g.pool(|| {
                let command_buffer = g.command_buffer();
                encode_tiny(g, s, &command_buffer, tiny);
                let (send, receive) = mpsc::channel();
                g.on_completed(
                    &command_buffer,
                    Box::new(move || {
                        let _ = send.send(Instant::now());
                    }),
                );
                let t0 = Instant::now();
                g.commit(&command_buffer);
                let at = receive.recv().expect("the completed handler ran");
                if run >= 10 {
                    handler.push(us(at - t0));
                }
            });
        }
        emit(
            "latency",
            &format!("{name}_commit_to_handler"),
            "us",
            &mut handler,
        );
    }
}

/// The `signals` workload (module docs of `main.rs`): a command buffer of one tiny fill with a
/// completed handler, committed alone and waited for, as the consumer's isolated execution is,
/// `plain`, and with what each other variant adds.
pub fn signals<G: Gpu>(g: &G, s: &Scene<G>, sizes: &Sizes) {
    for name in ["plain", "scheduled_handler", "scheduled_signal", "checked"] {
        let event = g.shared_event();
        let (mut handler, mut wait, mut cpu) = (vec![], vec![], vec![]);
        let mut counts = None;
        let mut has = true;
        let runs = sizes.latency + 10;
        for run in 0..runs {
            if run == 10 {
                counts = Some(g.work_counts());
            }
            let value = run as u64 + 1;
            g.pool(|| {
                let command_buffer = g.command_buffer();
                encode_tiny(g, s, &command_buffer, Tiny::Fill);
                match name {
                    "scheduled_handler" => g.set_when_scheduled(&command_buffer, &event, value),
                    "scheduled_signal" => g.signal_when_scheduled(&command_buffer, &event, value),
                    "checked" => has = g.check_submissions(&command_buffer),
                    _ => {}
                }
                let (send, receive) = mpsc::channel();
                g.on_completed(
                    &command_buffer,
                    Box::new(move || {
                        let _ = send.send(Instant::now());
                    }),
                );
                let cpu0 = process_cpu();
                let t0 = Instant::now();
                g.commit(&command_buffer);
                let at = receive.recv().expect("the completed handler ran");
                g.wait(&command_buffer);
                let t1 = Instant::now();
                let cpu1 = process_cpu();
                if run >= 10 {
                    handler.push(us(at - t0));
                    wait.push(us(t1 - t0));
                    cpu.push(us(cpu1 - cpu0));
                }
            });
            if !has {
                break;
            }
            if name.starts_with("scheduled") {
                assert_eq!(g.event_value(&event), value, "{name}: the value set");
            }
        }
        if !has {
            continue;
        }
        emit(
            "signals",
            &format!("{name}_commit_to_handler"),
            "us",
            &mut handler,
        );
        emit(
            "signals",
            &format!("{name}_commit_to_wait_return"),
            "us",
            &mut wait,
        );
        emit("signals", &format!("{name}_process_cpu"), "us", &mut cpu);
        let after = g.work_counts();
        for ((counted, before), (_, now)) in counts.unwrap_or_default().iter().zip(&after) {
            let per = (now - before) / sizes.latency as f64;
            emit(
                "signals",
                &format!("rt_{name}_{counted}"),
                "per cb",
                &mut [per],
            );
        }
    }
    // A burst of command buffers committed back to back, each checked or not, to the last one's
    // completion: the scheduler's cost of a batch, per command buffer.
    const BURST: usize = 64;
    for name in ["plain", "checked"] {
        let (mut wall, mut cpu) = (vec![], vec![]);
        let mut counts = None;
        let mut has = true;
        for run in 0..sizes.latency + 3 {
            if run == 3 {
                counts = Some(g.work_counts());
            }
            g.pool(|| {
                let buffers: Vec<_> = (0..BURST)
                    .map(|_| {
                        let command_buffer = g.command_buffer();
                        encode_tiny(g, s, &command_buffer, Tiny::Fill);
                        if name == "checked" {
                            has = g.check_submissions(&command_buffer);
                        }
                        command_buffer
                    })
                    .collect();
                let cpu0 = process_cpu();
                let t0 = Instant::now();
                for command_buffer in &buffers {
                    g.commit(command_buffer);
                }
                g.wait(&buffers[BURST - 1]);
                let t1 = Instant::now();
                let cpu1 = process_cpu();
                if run >= 3 {
                    wall.push(us(t1 - t0) / BURST as f64);
                    cpu.push(us(cpu1 - cpu0) / BURST as f64);
                }
            });
            if !has {
                break;
            }
        }
        if !has {
            continue;
        }
        emit("signals", &format!("burst_{name}_wall"), "us/cb", &mut wall);
        emit(
            "signals",
            &format!("burst_{name}_process_cpu"),
            "us/cb",
            &mut cpu,
        );
        let after = g.work_counts();
        for ((counted, before), (_, now)) in counts.unwrap_or_default().iter().zip(&after) {
            let per = (now - before) / (sizes.latency * BURST) as f64;
            emit(
                "signals",
                &format!("rt_burst_{name}_{counted}"),
                "per cb",
                &mut [per],
            );
        }
    }
    // A listener made and dropped unused, as the consumer makes one per task; then one made, and
    // its first notification asked for a value already reached, to its block's start.
    let (mut unused, mut made, mut first) = (vec![], vec![], vec![]);
    let event = g.shared_event();
    for run in 0..sizes.latency + 10 {
        let t0 = Instant::now();
        drop(g.listener());
        let t1 = Instant::now();
        let listener = g.listener();
        let t2 = Instant::now();
        let (send, receive) = mpsc::channel();
        g.notify(
            &event,
            &listener,
            0,
            Box::new(move || {
                let _ = send.send(Instant::now());
            }),
        );
        let at = receive.recv().expect("the notification ran");
        drop(listener);
        if run >= 10 {
            unused.push(us(t1 - t0));
            made.push(us(t2 - t1));
            first.push(us(at - t2));
        }
    }
    emit(
        "signals",
        "listener_made_and_dropped_unused",
        "us",
        &mut unused,
    );
    emit("signals", "listener_made", "us", &mut made);
    emit("signals", "listener_first_notification", "us", &mut first);
}

/// GPU time of the heavy shaders and the feature kernels, one command buffer in flight: from the
/// commit until the wait returned, and `GPUEndTime - GPUStartTime` where the side measures it on
/// the GPU. With `outputs`, each kernel's output after its last run goes to `<outputs>/<name>.bin`.
pub fn gpu<G: Gpu>(g: &G, s: &Scene<G>, sizes: &Sizes, outputs: Option<&Path>) {
    let run = |name: &str, encode: &dyn Fn(&G::CommandBuffer)| {
        let (mut wall, mut times) = (vec![], vec![]);
        for run in 0..sizes.heavy_runs + 3 {
            g.pool(|| {
                let command_buffer = g.command_buffer();
                encode(&command_buffer);
                let t0 = Instant::now();
                g.commit(&command_buffer);
                g.wait(&command_buffer);
                let t1 = Instant::now();
                if run >= 3 {
                    wall.push(ms(t1 - t0));
                    if let Some(seconds) = g.gpu_seconds(&command_buffer) {
                        times.push(seconds * 1e3);
                    }
                }
            });
        }
        emit("gpu", &format!("{name}_commit_to_wait"), "ms", &mut wall);
        if !times.is_empty() {
            emit("gpu", &format!("{name}_gpu_time"), "ms", &mut times);
        }
    };
    let keep = |name: &str| {
        if let Some(dir) = outputs {
            let mut bytes = vec![0u8; HEAVY_OUT_BYTES];
            g.read_buffer(&s.heavy_out, &mut bytes);
            std::fs::write(dir.join(format!("{name}.bin")), bytes).expect("write an output");
        }
    };
    run("heavy_fragment", &|command_buffer| {
        let mut encoder = g.render_encoder(command_buffer, &s.heavy_target, true);
        g.set_render_pipeline(&mut encoder, &s.heavy_fragment);
        g.set_vertex_buffer(&mut encoder, &s.full, 0, 0);
        g.set_fragment_buffer(&mut encoder, &s.params, 0, 0);
        g.set_fragment_texture(&mut encoder, &s.heavy_texture, 0);
        g.set_fragment_sampler(&mut encoder, &s.sampler, 0);
        g.draw_strip(&mut encoder, 4);
        g.end_render(encoder);
    });
    run("heavy_kernel", &|command_buffer| {
        let mut encoder = g.compute_encoder(command_buffer);
        g.set_compute_pipeline(&mut encoder, &s.heavy_kernel);
        g.set_buffer(&mut encoder, &s.heavy_out, 0, 0);
        g.set_buffer(&mut encoder, &s.kernel_params, 0, 1);
        g.set_texture(&mut encoder, &s.kernel_texture, 0);
        g.dispatch(&mut encoder, HEAVY_GROUPS, HEAVY_THREADS);
        g.end_compute(encoder);
    });
    keep("heavy_kernel");
    for (name, state) in &s.features {
        run(name, &|command_buffer| {
            let mut encoder = g.compute_encoder(command_buffer);
            g.set_compute_pipeline(&mut encoder, state);
            g.set_buffer(&mut encoder, &s.heavy_out, 0, 0);
            g.set_buffer(&mut encoder, &s.kernel_params, 0, 1);
            g.set_buffer(&mut encoder, &s.heavy_in, 0, 2);
            g.set_texture(&mut encoder, &s.kernel_texture, 0);
            g.dispatch(&mut encoder, HEAVY_GROUPS, HEAVY_THREADS);
            g.end_compute(encoder);
        });
        keep(name);
    }
}

/// Pipeline creation, in a process whose caches are as the caller left them: the library, each
/// state's creation, its first use (a command buffer of one draw or dispatch, to completion), and
/// each state made again in the same process. Single samples.
pub fn pipeline<G: Gpu>(g: &G, metallib: &std::path::Path, sizes: &Sizes) {
    let t = Instant::now();
    let library = g.library(metallib);
    emit("pipeline", "library", "ms", &mut [ms(t.elapsed())]);
    let target = g.texture(Format::Bgra8, TARGET_SIDE, TARGET_SIDE, true);
    let heavy_target = g.texture(Format::Rgba16Float, 256, 256, true);
    let side = KERNEL_TEXTURE_SIDE;
    let texture = g.texture(Format::Rgba8, side, side, false);
    g.replace_region(&texture, (side, side), &texels(side, 1), side * 4);
    let sampler = g.sampler(true);
    let rect = g.buffer_with(&floats(&[-1.0, -1.0, 1.0, 1.0]));
    let params = g.buffer_with(&params_bytes(sizes.heavy_iterations.min(4)));
    let out = g.buffer(HEAVY_THREADS * 16);
    let values = g.buffer(HEAVY_THREADS * 4);
    for pass in ["first", "again"] {
        let timed = |name: &str, f: &mut dyn FnMut()| {
            let t = Instant::now();
            f();
            emit(
                "pipeline",
                &format!("{name}_{pass}"),
                "ms",
                &mut [ms(t.elapsed())],
            );
        };
        let draw = |state: &G::RenderPipeline, target: &G::Texture, uniforms: bool| {
            g.pool(|| {
                let command_buffer = g.command_buffer();
                let mut encoder = g.render_encoder(&command_buffer, target, true);
                g.set_render_pipeline(&mut encoder, state);
                g.set_vertex_buffer(&mut encoder, &rect, 0, 0);
                if uniforms {
                    g.set_fragment_buffer(&mut encoder, &params, 0, 0);
                }
                g.set_fragment_texture(&mut encoder, &texture, 0);
                g.set_fragment_sampler(&mut encoder, &sampler, 0);
                g.draw_strip(&mut encoder, 4);
                g.end_render(encoder);
                g.commit(&command_buffer);
                g.wait(&command_buffer);
            })
        };
        let dispatch = |state: &G::ComputePipeline, heavy: bool| {
            g.pool(|| {
                let command_buffer = g.command_buffer();
                let mut encoder = g.compute_encoder(&command_buffer);
                g.set_compute_pipeline(&mut encoder, state);
                g.set_buffer(&mut encoder, &out, 0, 0);
                if heavy {
                    g.set_buffer(&mut encoder, &params, 0, 1);
                    g.set_texture(&mut encoder, &texture, 0);
                } else {
                    g.set_buffer(&mut encoder, &values, 0, 1);
                }
                g.dispatch(&mut encoder, 1, HEAVY_THREADS);
                g.end_compute(encoder);
                g.commit(&command_buffer);
                g.wait(&command_buffer);
            })
        };
        let mut state = None;
        timed("textured_create", &mut || {
            state = Some(g.render_pipeline(
                &library,
                "quad_vertex",
                "textured_fragment",
                Format::Bgra8,
                false,
            ))
        });
        let textured = state.take().unwrap();
        timed("textured_first_draw", &mut || {
            draw(&textured, &target, false)
        });
        timed("heavy_fragment_create", &mut || {
            state = Some(g.render_pipeline(
                &library,
                "quad_vertex",
                "heavy_fragment",
                Format::Rgba16Float,
                false,
            ))
        });
        let heavy = state.take().unwrap();
        timed("heavy_fragment_first_draw", &mut || {
            draw(&heavy, &heavy_target, true)
        });
        let mut kernel = None;
        timed("heavy_kernel_create", &mut || {
            kernel = Some(g.compute_pipeline(&library, "heavy_kernel"))
        });
        let heavy_kernel = kernel.take().unwrap();
        timed("heavy_kernel_first_dispatch", &mut || {
            dispatch(&heavy_kernel, true)
        });
        timed("tiny_kernel_create", &mut || {
            kernel = Some(g.compute_pipeline(&library, "tiny_kernel"))
        });
        let tiny = kernel.take().unwrap();
        timed("tiny_kernel_first_dispatch", &mut || dispatch(&tiny, false));
    }
}

/// Memory: buffer creation (`Shared`, by length, by bytes, and over host memory without a copy)
/// and a `Shared` texture's `replaceRegion` and `getBytes`; only the parts `variants` names where
/// it names any (`create`, `no_copy`, `held_1m`, `held_64m`, `texture`).
pub fn memory<G: Gpu>(g: &G, sizes: &Sizes, variants: &[String]) {
    const KIB: usize = 1024;
    const MIB: usize = 1024 * 1024;
    let runs = |name: &str| variants.is_empty() || variants.iter().any(|v| v == name);
    let repeats = |length: usize| {
        if length >= 16 * MIB {
            8
        } else {
            sizes.timed.max(20)
        }
    };
    let created = [("4k", 4 * KIB), ("1m", MIB), ("64m", 64 * MIB)];
    for (name, length) in created.into_iter().filter(|_| runs("create")) {
        let mut times = vec![];
        for _ in 0..repeats(length) {
            let t = Instant::now();
            let buffer = g.buffer(length);
            times.push(us(t.elapsed()));
            drop(buffer);
        }
        emit("memory", &format!("new_buffer_{name}"), "us", &mut times);
        let bytes = vec![0x33u8; length];
        let mut times = vec![];
        for _ in 0..repeats(length) {
            let t = Instant::now();
            let buffer = g.buffer_with(&bytes);
            times.push(us(t.elapsed()));
            drop(buffer);
        }
        emit(
            "memory",
            &format!("new_buffer_with_bytes_{name}"),
            "us",
            &mut times,
        );
    }
    let no_copy = [("1m", MIB), ("64m", 64 * MIB)];
    for (name, length) in no_copy.into_iter().filter(|_| runs("no_copy")) {
        let mut times = vec![];
        let mut maps = vec![];
        for _ in 0..repeats(length) {
            // SAFETY: an anonymous private mapping of `length` bytes, unmapped below once every
            // buffer over it is gone.
            let pointer = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    length,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_ANON | libc::MAP_PRIVATE,
                    -1,
                    0,
                )
            };
            assert_ne!(pointer, libc::MAP_FAILED);
            // SAFETY: the pages are mapped read-write; touch them as a program's own data is.
            unsafe { std::ptr::write_bytes(pointer as *mut u8, 0x11, length) };
            let t = Instant::now();
            // SAFETY: page-aligned read-write memory that outlives the buffer (unmapped below).
            let buffer = unsafe { g.buffer_no_copy(pointer as *mut u8, length) };
            times.push(us(t.elapsed()));
            drop(buffer);
            maps.push(pointer);
        }
        emit(
            "memory",
            &format!("new_buffer_no_copy_{name}"),
            "us",
            &mut times,
        );
        // Let any release the side defers finish before the pages go.
        let command_buffer = g.command_buffer();
        g.commit(&command_buffer);
        g.wait(&command_buffer);
        std::thread::sleep(Duration::from_millis(50));
        for pointer in maps {
            // SAFETY: mapped above, no buffer over it remains.
            unsafe { libc::munmap(pointer, length) };
        }
    }
    // Buffers held, as a program that grows its memory makes them: each takes memory no buffer
    // of the run used before, so neither side hands back one it made earlier (the runtime's
    // recycler lets go of what it parked above once a second passed). Each one's creation, then
    // the first touch of all its bytes: a write from the CPU, or, of another set, a blit fill
    // from the GPU (commit to the wait's return), and that fill again.
    let held = [("1m", MIB, 48), ("64m", 64 * MIB, 6)];
    for (name, length, count) in held
        .into_iter()
        .filter(|(name, ..)| runs(&format!("held_{name}")))
    {
        std::thread::sleep(Duration::from_millis(1100));
        let mut make = vec![];
        let mut held = || -> Vec<G::Buffer> {
            (0..count)
                .map(|_| {
                    let t = Instant::now();
                    let buffer = g.buffer(length);
                    make.push(us(t.elapsed()));
                    buffer
                })
                .collect()
        };
        // Both sets held to the end, so that the second takes none of the first's memory.
        let (written, filled) = (held(), held());
        let mut write = vec![];
        for buffer in &written {
            let t = Instant::now();
            g.write_buffer(buffer, 0x5a);
            write.push(us(t.elapsed()));
        }
        let (mut fill, mut again) = (vec![], vec![]);
        for buffer in &filled {
            for times in [&mut fill, &mut again] {
                let t = Instant::now();
                let command_buffer = g.command_buffer();
                g.fill(&command_buffer, buffer, length);
                g.commit(&command_buffer);
                g.wait(&command_buffer);
                times.push(us(t.elapsed()));
            }
        }
        drop((written, filled));
        emit(
            "memory",
            &format!("new_buffer_held_{name}"),
            "us",
            &mut make,
        );
        emit(
            "memory",
            &format!("first_write_held_{name}"),
            "us",
            &mut write,
        );
        emit(
            "memory",
            &format!("first_fill_held_{name}"),
            "us",
            &mut fill,
        );
        emit(
            "memory",
            &format!("second_fill_held_{name}"),
            "us",
            &mut again,
        );
    }
    if !runs("texture") {
        return;
    }
    let side = HEAVY_TEXTURE_SIDE;
    let texture = g.texture(Format::Rgba8, side, side, false);
    let bytes = texels(side, 3);
    let mut back = vec![0u8; bytes.len()];
    let (mut replace, mut get) = (vec![], vec![]);
    for run in 0..sizes.timed.max(20) + 2 {
        let t0 = Instant::now();
        g.replace_region(&texture, (side, side), &bytes, side * 4);
        let t1 = Instant::now();
        g.get_bytes(&texture, (side, side), &mut back, side * 4);
        let t2 = Instant::now();
        if run >= 2 {
            replace.push(us(t1 - t0));
            get.push(us(t2 - t1));
        }
    }
    assert!(back == bytes, "getBytes read back what replaceRegion wrote");
    emit("memory", "texture_replace_region_4m", "us", &mut replace);
    emit("memory", "texture_get_bytes_4m", "us", &mut get);
}

/// The `frame` workload's textures, their side, and the bytes between their offsets.
const FRAME_TEXTURES: usize = 8;
const FRAME_SIDE: usize = 256;
const FRAME_STRIDE: usize = 1 << 20;

/// One frame of the `frame` workload (module docs of `main.rs`) in `heap`: the textures made,
/// each rendered by a pass of three draws, each sampled into the target, committed, waited for
/// and dropped; the times of each step into `times` (`[encode, make, drop, process CPU]`, the CPU
/// per frame).
fn frame_once<G: Gpu>(g: &G, s: &Scene<G>, heap: &G::Heap, times: &mut [Vec<f64>; 4]) {
    g.pool(|| {
        let cpu0 = process_cpu();
        let t0 = Instant::now();
        let textures: Vec<G::Texture> = (0..FRAME_TEXTURES)
            .map(|i| g.heap_texture(heap, FRAME_SIDE, FRAME_SIDE, i * FRAME_STRIDE))
            .collect();
        let t1 = Instant::now();
        let command_buffer = g.command_buffer();
        for (i, texture) in textures.iter().enumerate() {
            let mut encoder = g.render_encoder(&command_buffer, texture, true);
            g.set_render_pipeline(&mut encoder, &s.textured[0]);
            g.set_fragment_sampler(&mut encoder, &s.sampler, 0);
            for draw in 0..PASS_DRAWS {
                let rect = (i * PASS_DRAWS + draw) % RECTS;
                g.set_vertex_buffer(&mut encoder, &s.rects[0], rect * RECT_STRIDE, 0);
                g.set_fragment_texture(&mut encoder, &s.textures[(i + draw) % TEXTURES], 0);
                g.draw_strip(&mut encoder, 4);
            }
            g.end_render(encoder);
        }
        let mut encoder = g.render_encoder(&command_buffer, &s.target, true);
        g.set_render_pipeline(&mut encoder, &s.textured[1]);
        g.set_fragment_sampler(&mut encoder, &s.sampler, 0);
        for (i, texture) in textures.iter().enumerate() {
            g.set_vertex_buffer(&mut encoder, &s.rects[1], i * RECT_STRIDE, 0);
            g.set_fragment_texture(&mut encoder, texture, 0);
            g.draw_strip(&mut encoder, 4);
        }
        g.end_render(encoder);
        g.commit(&command_buffer);
        let t2 = Instant::now();
        g.wait(&command_buffer);
        let t3 = Instant::now();
        drop(textures);
        let t4 = Instant::now();
        let cpu1 = process_cpu();
        times[0].push(us(t2 - t0));
        times[1].push(us(t1 - t0));
        times[2].push(us(t4 - t3));
        times[3].push(us(cpu1 - cpu0));
    });
}

/// A window server's frame of heap textures (module docs of `main.rs`), in a heap of the device,
/// and the runtime's over host pages.
pub fn frame<G: Gpu>(g: &G, s: &Scene<G>, sizes: &Sizes) {
    let size = FRAME_TEXTURES * FRAME_STRIDE;
    let run = |name: &str, heap: &G::Heap| {
        let mut times: [Vec<f64>; 4] = Default::default();
        for run in 0..sizes.timed + 3 {
            let mut taken: [Vec<f64>; 4] = Default::default();
            frame_once(g, s, heap, &mut taken);
            if run >= 3 {
                for (all, one) in times.iter_mut().zip(taken) {
                    all.extend(one);
                }
            }
        }
        let [encode, make, dropped, cpu] = &mut times;
        emit("frame", &format!("{name}_encode"), "us", encode);
        emit("frame", &format!("{name}_make_textures"), "us", make);
        emit("frame", &format!("{name}_drop_textures"), "us", dropped);
        emit("frame", &format!("{name}_process_cpu"), "us", cpu);
    };
    run("frame", &g.heap(size));
    // SAFETY: an anonymous private mapping of `size` bytes, never unmapped: the process's to its
    // end, past the heap and every texture placed in it.
    let pointer = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_ANON | libc::MAP_PRIVATE,
            -1,
            0,
        )
    };
    assert_ne!(pointer, libc::MAP_FAILED);
    // SAFETY: page-aligned read-write memory of `size` bytes that outlives the heap.
    if let Some(heap) = unsafe { g.heap_no_copy(pointer as *mut u8, size) } {
        run("frame_no_copy", &heap);
    }
}

/// The `drag` workload's frame (module docs of `main.rs`): its command buffers, its render passes
/// (over [`DRAG_SURFACES`] window surfaces), the draws of each pass, and its dispatches.
const DRAG_COMMAND_BUFFERS: usize = 40;
const DRAG_PASSES: usize = 31;
const DRAG_PASS_DRAWS: usize = 3;
const DRAG_DISPATCHES: usize = 22;
const DRAG_SURFACES: usize = 8;

/// The bytes a window surface's rows lie apart in the `drag` workload's surfaces over host pages:
/// its packed row padded, as a surface's rows are.
const DRAG_SURFACE_ROW: usize = FRAME_SIDE * 4 + 256;

/// One frame of the `drag` workload: [`DRAG_COMMAND_BUFFERS`] command buffers, pass `p` in
/// command buffer `p` and a dispatch in each of the first [`DRAG_DISPATCHES`], each committed;
/// the last waited for. A pass's draws sample the scene's textures, or, where the frame
/// composites (`uploads` holds the windows' contents), the other surfaces, as a compositor's do,
/// the first command buffer uploading `uploads` into half the surfaces in one blit encoder before
/// its pass. Its times into `times` (`[encode, frame, thread CPU, process CPU]`).
fn drag_once<G: Gpu>(
    g: &G,
    s: &Scene<G>,
    (surfaces, uploads): (&[G::Texture], Option<&G::Buffer>),
    times: &mut [Vec<f64>; 4],
) {
    g.pool(|| {
        let cpu = process_cpu();
        let thread = thread_cpu();
        let start = Instant::now();
        let mut last = None;
        for at in 0..DRAG_COMMAND_BUFFERS {
            let command_buffer = g.command_buffer();
            if let Some(contents) = uploads.filter(|_| at == 0) {
                let windows: Vec<&G::Texture> = surfaces.iter().step_by(2).collect();
                let side = (FRAME_SIDE, FRAME_SIDE);
                g.upload(&command_buffer, contents, &windows, side, FRAME_SIDE * 4);
            }
            if at < DRAG_PASSES {
                let target = &surfaces[at % DRAG_SURFACES];
                let mut encoder = g.render_encoder(&command_buffer, target, at % 4 != 0);
                g.set_render_pipeline(&mut encoder, &s.textured[at % 2]);
                g.set_fragment_sampler(&mut encoder, &s.sampler, 0);
                for draw in 0..DRAG_PASS_DRAWS {
                    let rect = (at * DRAG_PASS_DRAWS + draw) % RECTS;
                    g.set_vertex_buffer(&mut encoder, &s.rects[0], rect * RECT_STRIDE, 0);
                    let texture = match uploads {
                        Some(_) => &surfaces[(at + draw + 1) % DRAG_SURFACES],
                        None => &s.textures[(at + draw) % TEXTURES],
                    };
                    g.set_fragment_texture(&mut encoder, texture, 0);
                    g.draw_strip(&mut encoder, 4);
                }
                g.end_render(encoder);
            }
            if at < DRAG_DISPATCHES {
                encode_dispatches(g, s, &command_buffer, 1, DispatchChurn::Static);
            }
            g.commit(&command_buffer);
            last = Some(command_buffer);
        }
        let encoded = Instant::now();
        let thread = thread_cpu() - thread;
        if let Some(command_buffer) = &last {
            g.wait(command_buffer);
        }
        let done = Instant::now();
        let cpu = process_cpu() - cpu;
        times[0].push(ms(encoded - start));
        times[1].push(ms(done - start));
        times[2].push(ms(thread));
        times[3].push(ms(cpu));
    });
}

/// A frame shaped as the guest's window drag (module docs of `main.rs`), its window surfaces in
/// a heap of the device, and the runtime's in a heap over host pages too; then its surfaces as
/// textures over host pages, one buffer each, the draws sampling the other surfaces.
pub fn drag<G: Gpu>(g: &G, s: &Scene<G>, sizes: &Sizes) {
    let size = DRAG_SURFACES * FRAME_STRIDE;
    let contents = g.buffer(FRAME_SIDE * FRAME_SIDE * 4);
    let run = |name: &str, surfaces: &[G::Texture], composites: bool| {
        let uploads = composites.then_some(&contents);
        let mut times: [Vec<f64>; 4] = Default::default();
        for frame in 0..sizes.timed + 3 {
            let mut taken: [Vec<f64>; 4] = Default::default();
            drag_once(g, s, (surfaces, uploads), &mut taken);
            if frame >= 3 {
                for (all, one) in times.iter_mut().zip(taken) {
                    all.extend(one);
                }
            }
        }
        let [encode, frame, thread, cpu] = &mut times;
        emit("drag", &format!("{name}_encode"), "ms", encode);
        emit("drag", &format!("{name}_frame"), "ms", frame);
        emit("drag", &format!("{name}_thread_cpu"), "ms", thread);
        emit("drag", &format!("{name}_process_cpu"), "ms", cpu);
    };
    let in_heap = |heap: &G::Heap| -> Vec<G::Texture> {
        (0..DRAG_SURFACES)
            .map(|i| g.heap_texture(heap, FRAME_SIDE, FRAME_SIDE, i * FRAME_STRIDE))
            .collect()
    };
    run("drag", &in_heap(&g.heap(size)), false);
    // SAFETY: an anonymous private mapping of `size` bytes, never unmapped: the process's to its
    // end, past the heap and every texture placed in it.
    let pointer = unsafe { host_pages(size) };
    // SAFETY: page-aligned read-write memory of `size` bytes that outlives the heap.
    if let Some(heap) = unsafe { g.heap_no_copy(pointer, size) } {
        run("drag_no_copy", &in_heap(&heap), false);
    }
    // SAFETY: as above, a mapping of its own.
    let pointer = unsafe { host_pages(size) };
    // SAFETY: page-aligned read-write memory of `size` bytes that outlives the heap.
    if let Some(heap) = unsafe { g.heap_no_copy(pointer, size) } {
        let surfaces: Vec<G::Texture> = (0..DRAG_SURFACES)
            .map(|i| g.heap_render_texture(&heap, Format::Bgra8, FRAME_SIDE, i * FRAME_STRIDE))
            .collect();
        run("drag_heap_surfaces", &surfaces, true);
    }
    let surfaces: Vec<G::Texture> = (0..DRAG_SURFACES)
        .map(|_| {
            // SAFETY: as above, a mapping of its own for each surface.
            let buffer = unsafe { g.buffer_no_copy(host_pages(FRAME_STRIDE), FRAME_STRIDE) };
            let side = (FRAME_SIDE, FRAME_SIDE);
            g.buffer_texture(&buffer, side, 0, DRAG_SURFACE_ROW)
        })
        .collect();
    run("drag_surfaces", &surfaces, true);
}

/// An anonymous private mapping of `size` bytes, read-write.
///
/// # Safety
///
/// Never unmapped: the caller keeps it for the process's life.
pub unsafe fn host_pages(size: usize) -> *mut u8 {
    // SAFETY: a new anonymous mapping, which nothing else holds.
    let pointer = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_ANON | libc::MAP_PRIVATE,
            -1,
            0,
        )
    };
    assert_ne!(pointer, libc::MAP_FAILED);
    pointer as *mut u8
}

/// The many-draws workload's target (1080p `BGRA8Unorm`, with a `Depth32Float` attachment), its
/// textures (16 `RGBA8Unorm` of 128x128) and its uniforms' stride.
const MANY_TARGET: (usize, usize) = (1920, 1080);
const MANY_TEXTURES: usize = 16;
const MANY_TEXTURE_SIDE: usize = 128;
const MANY_STRIDE: usize = 256;

/// What the many-draws workload draws with (module docs of `main.rs`).
struct ManyDraws<G: Gpu> {
    /// `draws_vertex` with each of `draws_fragment0` to `3`, the third blending.
    pipelines: [G::RenderPipeline; 4],
    /// `Less` with depth writes, `LessEqual` without.
    depth_states: [G::DepthStencil; 2],
    /// Linear and nearest.
    samplers: [G::Sampler; 2],
    textures: Vec<G::Texture>,
    /// The rectangle's corners, and its two triangles' indices.
    quad: G::Buffer,
    indices: G::Buffer,
    /// Four uniform buffers: draw `i` reads buffer `i % 4` at `(i / 4) * MANY_STRIDE`.
    uniforms: [G::Buffer; 4],
    /// Where the uniforms lie in host pages: the heap over those pages, which the buffers lie in
    /// too, as a guest's buffers lie in its memory beside its heaps.
    _pages: Option<G::Heap>,
    target: G::Texture,
    depth: G::Texture,
}

impl<G: Gpu> ManyDraws<G> {
    /// The scene of `count` draws, its uniforms in buffers of their own, or with `pages` in
    /// no-copy buffers over host pages a no-copy heap lies over too: `None` where the side makes
    /// no such heap.
    fn new(g: &G, library: &G::Library, count: usize, pages: bool) -> Option<ManyDraws<G>> {
        let pipeline = |i: usize| {
            g.depth_pipeline(
                library,
                "draws_vertex",
                &format!("draws_fragment{i}"),
                i == 2,
            )
        };
        let textures = (0..MANY_TEXTURES)
            .map(|n| {
                let side = MANY_TEXTURE_SIDE;
                let texture = g.texture(Format::Rgba8, side, side, false);
                g.replace_region(
                    &texture,
                    (side, side),
                    &texels(side, n as u8 * 13 + 1),
                    side * 4,
                );
                texture
            })
            .collect();
        // A grid of rectangles over the target, each half again a cell's side, as m2vbench lays
        // them.
        let grid = (count as f64).sqrt().ceil() as usize;
        let cell = 2.0 / grid as f32;
        let per = count.div_ceil(4) * MANY_STRIDE;
        let mut bytes = [(); 4].map(|_| vec![0u8; per]);
        for i in 0..count {
            let (x, y) = ((i % grid) as f32, (i / grid) as f32);
            let at = (i / 4) * MANY_STRIDE;
            let uniforms = floats(&[
                -1.0 + x * cell,
                -1.0 + y * cell,
                cell * 1.5,
                cell * 1.5,
                0.5 + 0.5 * (i as f32 * 0.1).sin(),
                0.5 + 0.5 * (i as f32 * 0.07).cos(),
                (i % 7) as f32 / 7.0,
                1.0,
                i as f32 / count as f32,
                i as f32 * 0.001,
                0.0,
                0.0,
            ]);
            bytes[i % 4][at..at + uniforms.len()].copy_from_slice(&uniforms);
        }
        let indices: Vec<u8> = [0u16, 1, 2, 2, 1, 3]
            .iter()
            .flat_map(|i| i.to_le_bytes())
            .collect();
        let (uniforms, pages) = match pages {
            false => (bytes.map(|bytes| g.buffer_with(&bytes)), None),
            true => {
                let page = 16 << 10;
                let each = per.next_multiple_of(page);
                // SAFETY: a new mapping of four buffers' pages, never unmapped: the process's to
                // its end, past the heap and the buffers over it.
                let pointer = unsafe { host_pages(4 * each) };
                for (at, bytes) in bytes.iter().enumerate() {
                    // SAFETY: `at * each + per` bytes lie in the mapping, which nothing uses yet.
                    unsafe {
                        std::ptr::copy_nonoverlapping(bytes.as_ptr(), pointer.add(at * each), per)
                    };
                }
                // SAFETY: page-aligned read-write memory that outlives the heap.
                let heap = unsafe { g.heap_no_copy(pointer, 4 * each) }?;
                // SAFETY: as above, each buffer over its own pages of the mapping.
                let buffer = |at: usize| unsafe { g.buffer_no_copy(pointer.add(at * each), each) };
                ([0, 1, 2, 3].map(buffer), Some(heap))
            }
        };
        Some(ManyDraws {
            pipelines: [pipeline(0), pipeline(1), pipeline(2), pipeline(3)],
            depth_states: [g.depth_stencil(true), g.depth_stencil(false)],
            samplers: [g.sampler(true), g.sampler(false)],
            textures,
            quad: g.buffer_with(&floats(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])),
            indices: g.buffer_with(&indices),
            uniforms,
            _pages: pages,
            target: g.texture(Format::Bgra8, MANY_TARGET.0, MANY_TARGET.1, true),
            depth: g.texture(Format::Depth32Float, MANY_TARGET.0, MANY_TARGET.1, true),
        })
    }

    /// Encode one frame's render pass of `count` draws into `command_buffer`, as m2vbench's draws
    /// mode does.
    fn encode(&self, g: &G, command_buffer: &G::CommandBuffer, count: usize) {
        let mut encoder = g.depth_render_encoder(command_buffer, &self.target, &self.depth);
        g.set_vertex_buffer(&mut encoder, &self.quad, 0, 0);
        g.set_counter_clockwise(&mut encoder);
        for i in 0..count {
            if i % 8 == 0 {
                g.set_render_pipeline(&mut encoder, &self.pipelines[(i / 8) % 4]);
            }
            if i % 16 == 0 {
                g.set_depth_stencil(&mut encoder, &self.depth_states[(i / 16) % 2]);
            }
            if i % 32 == 0 {
                g.set_fragment_sampler(&mut encoder, &self.samplers[(i / 32) % 2], 0);
            }
            if i % 64 == 0 {
                g.set_cull_back(&mut encoder, (i / 64) % 2 == 0);
            }
            let uniforms = &self.uniforms[i % 4];
            let at = (i / 4) * MANY_STRIDE;
            g.set_vertex_buffer(&mut encoder, uniforms, at, 1);
            g.set_fragment_buffer(&mut encoder, uniforms, at, 0);
            g.set_fragment_texture(&mut encoder, &self.textures[i % MANY_TEXTURES], 0);
            g.set_fragment_texture(&mut encoder, &self.textures[(i * 7 + 3) % MANY_TEXTURES], 1);
            let mut jitter = [0u8; 16];
            jitter[..4].copy_from_slice(&(0.002 * (i % 3) as f32).to_le_bytes());
            g.set_vertex_bytes(&mut encoder, &jitter, 2);
            if i % 2 == 1 {
                g.draw_indexed(&mut encoder, 6, &self.indices);
            } else {
                g.draw_strip(&mut encoder, 4);
            }
        }
        g.end_render(encoder);
    }
}

/// The many-draws workload (module docs of `main.rs`): per frame, the encoding thread's wall
/// time from the command buffer's creation through `endEncoding`, the commit call's, the commit's
/// return to the wait's, the whole frame's, the process's CPU time, the encoding thread's CPU
/// time through the commit's return, and the GPU time where the side measures it.
pub fn many_draws<G: Gpu>(g: &G, library: &G::Library, sizes: &Sizes) {
    let count = sizes.many_draws;
    for (name, pages) in [("", false), ("pages_", true)] {
        let Some(scene) = ManyDraws::new(g, library, count, pages) else {
            continue;
        };
        many_draws_of(g, &scene, (name, count), sizes);
    }
}

/// The many-draws workload's frames of `scene`, `count` draws each, its rows' names prefixed with
/// `name`.
fn many_draws_of<G: Gpu>(g: &G, scene: &ManyDraws<G>, (name, count): (&str, usize), sizes: &Sizes) {
    let mut times: [Vec<f64>; 7] = Default::default();
    for frame in 0..sizes.timed + 3 {
        g.pool(|| {
            let cpu = process_cpu();
            let thread = thread_cpu();
            let start = Instant::now();
            let command_buffer = g.command_buffer();
            scene.encode(g, &command_buffer, count);
            let encoded = Instant::now();
            g.commit(&command_buffer);
            let committed = Instant::now();
            let thread = thread_cpu() - thread;
            g.wait(&command_buffer);
            let done = Instant::now();
            let cpu = process_cpu() - cpu;
            if frame < 3 {
                return;
            }
            times[6].push(ms(thread));
            times[0].push(ms(encoded - start));
            times[1].push(ms(committed - encoded));
            times[2].push(ms(done - committed));
            times[3].push(ms(done - start));
            times[4].push(ms(cpu));
            if let Some(gpu) = g.gpu_seconds(&command_buffer) {
                times[5].push(gpu * 1e3);
            }
        });
    }
    let [encode, commit, wait, frame, cpu, gpu, thread] = &mut times;
    emit("many_draws", &format!("{name}encode"), "ms", encode);
    emit("many_draws", &format!("{name}commit"), "ms", commit);
    emit("many_draws", &format!("{name}commit_to_done"), "ms", wait);
    emit("many_draws", &format!("{name}frame"), "ms", frame);
    emit("many_draws", &format!("{name}cpu"), "ms", cpu);
    emit("many_draws", &format!("{name}thread_cpu"), "ms", thread);
    if !gpu.is_empty() {
        emit("many_draws", &format!("{name}gpu"), "ms", gpu);
    }
}

/// The layers workload's layers, its textures, their sides, and the target's size (m2vbench's
/// composite mode at its default count).
const LAYERS: usize = 800;
const LAYER_TEXTURES: usize = 48;
const LAYER_SIDES: [usize; 5] = [64, 96, 128, 192, 256];
const LAYERS_TARGET: (usize, usize) = (1920, 1080);

/// What the layers workload draws with (module docs of `main.rs`).
struct Layers<G: Gpu> {
    /// The layers' and the shadows' pipeline states; the wallpaper is drawn as a layer.
    layer: G::RenderPipeline,
    shadow: G::RenderPipeline,
    sampler: G::Sampler,
    wallpaper: G::Texture,
    textures: Vec<G::Texture>,
    /// Each layer's rectangle at rest: x, y, w, h in pixels.
    rests: Vec<[f32; 4]>,
    target: G::Texture,
}

impl<G: Gpu> Layers<G> {
    fn new(g: &G, library: &G::Library) -> Layers<G> {
        let (width, height) = LAYERS_TARGET;
        let pipeline =
            |fragment| g.render_pipeline(library, "layer_vertex", fragment, Format::Bgra8, true);
        let texture = |w: usize, h: usize, seed: usize| {
            let texture = g.texture(Format::Rgba8, w, h, false);
            let texels: Vec<u8> = (0..w * h)
                .flat_map(|i| {
                    let (x, y) = (i % w, i / w);
                    let v = ((x * 7 + y * 13) ^ seed.wrapping_mul(2_654_435_761)) as u8;
                    [
                        v,
                        (x * 255 / w) as u8,
                        ((y * 255 / h + seed * 31) % 256) as u8,
                        255,
                    ]
                })
                .collect();
            g.replace_region(&texture, (w, h), &texels, w * 4);
            texture
        };
        let textures = (0..LAYER_TEXTURES)
            .map(|i| texture(LAYER_SIDES[i % 5], LAYER_SIDES[(i / 5) % 5], i + 300))
            .collect();
        let mut seed = 42u32;
        let mut next = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        let rests = (0..LAYERS)
            .map(|_| {
                let (w, h) = (120.0 + next() * 560.0, 90.0 + next() * 420.0);
                let (x, y) = (next() * (width as f32 - w), next() * (height as f32 - h));
                [x, y, w, h]
            })
            .collect();
        Layers {
            layer: pipeline("layer_fragment"),
            shadow: pipeline("shadow_fragment"),
            sampler: g.sampler(true),
            wallpaper: texture(256, 256, 999),
            textures,
            rests,
            target: g.texture(Format::Bgra8, width, height, true),
        }
    }

    /// Encode one frame at time `t` into `command_buffer`, as m2vbench's composite mode does: the
    /// wallpaper, then each layer back to front, a shadow under every eighth, one draw each, its
    /// uniforms as vertex bytes and its texture bound before it.
    fn encode(&self, g: &G, command_buffer: &G::CommandBuffer, t: f32) {
        let (width, height) = (LAYERS_TARGET.0 as f32, LAYERS_TARGET.1 as f32);
        let uniforms = |rect: [f32; 4], opacity: f32| {
            floats(&[
                rect[0], rect[1], rect[2], rect[3], width, height, opacity, 12.0,
            ])
        };
        let mut encoder = g.render_encoder(command_buffer, &self.target, true);
        g.set_fragment_sampler(&mut encoder, &self.sampler, 0);
        g.set_render_pipeline(&mut encoder, &self.layer);
        g.set_vertex_bytes(&mut encoder, &uniforms([0.0, 0.0, width, height], 1.0), 0);
        g.set_fragment_texture(&mut encoder, &self.wallpaper, 0);
        g.draw_strip(&mut encoder, 4);
        for (i, rest) in self.rests.iter().enumerate() {
            let f = i as f32;
            let rect = [
                rest[0] + 40.0 * (t * 0.7 + f).sin(),
                rest[1] + 30.0 * (t * 0.5 + f * 0.3).cos(),
                rest[2],
                rest[3],
            ];
            let opacity = 0.75 + 0.25 * (f * 1.7).sin();
            if i % 8 == 0 {
                let grown = [
                    rect[0] - 32.0,
                    rect[1] - 24.0,
                    rect[2] + 64.0,
                    rect[3] + 64.0,
                ];
                g.set_render_pipeline(&mut encoder, &self.shadow);
                g.set_vertex_bytes(&mut encoder, &uniforms(grown, opacity), 0);
                g.draw_strip(&mut encoder, 4);
                g.set_render_pipeline(&mut encoder, &self.layer);
            }
            g.set_vertex_bytes(&mut encoder, &uniforms(rect, opacity), 0);
            g.set_fragment_texture(&mut encoder, &self.textures[i % LAYER_TEXTURES], 0);
            g.draw_strip(&mut encoder, 4);
        }
        g.end_render(encoder);
    }
}

/// The layers workload (module docs of `main.rs`): per frame, the encoding thread's wall time
/// through `endEncoding`, the commit call's, the commit's return to the wait's, the whole frame's,
/// the process's CPU time, the encoding thread's CPU time through the commit's return, and the
/// GPU time where the side measures it.
pub fn layers<G: Gpu>(g: &G, library: &G::Library, sizes: &Sizes) {
    let scene = Layers::new(g, library);
    let mut times: [Vec<f64>; 7] = Default::default();
    for frame in 0..sizes.timed + 3 {
        g.pool(|| {
            let cpu = process_cpu();
            let thread = thread_cpu();
            let start = Instant::now();
            let command_buffer = g.command_buffer();
            scene.encode(g, &command_buffer, frame as f32 / 60.0);
            let encoded = Instant::now();
            g.commit(&command_buffer);
            let committed = Instant::now();
            let thread = thread_cpu() - thread;
            g.wait(&command_buffer);
            let done = Instant::now();
            let cpu = process_cpu() - cpu;
            if frame < 3 {
                return;
            }
            times[6].push(ms(thread));
            times[0].push(ms(encoded - start));
            times[1].push(ms(committed - encoded));
            times[2].push(ms(done - committed));
            times[3].push(ms(done - start));
            times[4].push(ms(cpu));
            if let Some(gpu) = g.gpu_seconds(&command_buffer) {
                times[5].push(gpu * 1e3);
            }
        });
    }
    let [encode, commit, wait, frame, cpu, gpu, thread] = &mut times;
    emit("layers", "encode", "ms", encode);
    emit("layers", "commit", "ms", commit);
    emit("layers", "commit_to_done", "ms", wait);
    emit("layers", "frame", "ms", frame);
    emit("layers", "cpu", "ms", cpu);
    emit("layers", "thread_cpu", "ms", thread);
    if !gpu.is_empty() {
        emit("layers", "gpu", "ms", gpu);
    }
}

/// The textures each frame of the `churn` workload makes.
const CHURNED: usize = 8;
/// The buffers each frame of the `churn` workload's `buffers` makes, and their length.
const CHURNED_BUFFERS: usize = 16;
const CHURNED_LENGTH: usize = 64 << 10;

/// Churn: each frame makes eight 1280x800 `BGRA8Unorm` `Private` render targets, clears each in a
/// render pass of one command buffer, waits for it and releases them, as a window server's frame
/// makes and releases its textures. `same`: every frame's textures are of the same size; `varied`:
/// each frame's are a pixel wider than the frame before's, over 64 widths, so no texture's making
/// meets one of its shape released within the last 64 frames. `buffers`: each frame makes sixteen
/// 64-KiB buffers, fills each in a blit of one command buffer, waits for it and releases them, as
/// the staging and scratch buffers a frame's copies make. Per frame: the time from the first
/// making to the wait's return, and the process's CPU time.
pub fn churn<G: Gpu>(g: &G, sizes: &Sizes) {
    let frames = sizes.timed.max(60);
    for (name, varied) in [("same", false), ("varied", true)] {
        let (mut frame, mut cpu) = (vec![], vec![]);
        for i in 0..frames + 5 {
            let width = 1280 + if varied { i % 64 } else { 0 };
            g.pool(|| {
                let (t, c) = (Instant::now(), process_cpu());
                let textures: Vec<G::Texture> = (0..CHURNED)
                    .map(|_| g.texture(Format::Bgra8, width, 800, true))
                    .collect();
                let command_buffer = g.command_buffer();
                for texture in &textures {
                    let encoder = g.render_encoder(&command_buffer, texture, true);
                    g.end_render(encoder);
                }
                g.commit(&command_buffer);
                g.wait(&command_buffer);
                drop(command_buffer);
                drop(textures);
                if i >= 5 {
                    frame.push(ms(t.elapsed()));
                    cpu.push(ms(process_cpu() - c));
                }
            });
        }
        emit("churn", &format!("{name}_frame"), "ms", &mut frame);
        emit("churn", &format!("{name}_cpu"), "ms", &mut cpu);
    }
    let (mut frame, mut cpu) = (vec![], vec![]);
    for i in 0..frames + 5 {
        g.pool(|| {
            let (t, c) = (Instant::now(), process_cpu());
            let buffers: Vec<G::Buffer> = (0..CHURNED_BUFFERS)
                .map(|_| g.buffer(CHURNED_LENGTH))
                .collect();
            let command_buffer = g.command_buffer();
            for buffer in &buffers {
                g.fill(&command_buffer, buffer, CHURNED_LENGTH);
            }
            g.commit(&command_buffer);
            g.wait(&command_buffer);
            drop(command_buffer);
            drop(buffers);
            if i >= 5 {
                frame.push(ms(t.elapsed()));
                cpu.push(ms(process_cpu() - c));
            }
        });
    }
    emit("churn", "buffers_frame", "ms", &mut frame);
    emit("churn", "buffers_cpu", "ms", &mut cpu);
}

/// The compositing workload's command buffers a frame, and the new heap textures each frame
/// makes.
const COMPOSITING_BUFFERS: usize = 20;
const COMPOSITING_NEW: usize = 10;
/// A new texture's side, and the heap's bytes for each (past any texture's alignment).
const COMPOSITING_NEW_SIDE: usize = 256;
const COMPOSITING_SLOT: usize = 1 << 20;
/// The two layers' and the screen's side.
const COMPOSITING_SIDE: usize = 1024;
/// Frames run before the timed ones.
const COMPOSITING_WARM: usize = 20;

/// Frames of the consumer's compositing shape (module docs of `main.rs`), paced at 60 Hz: per
/// frame, its wall time from the first command buffer's creation to the last one's completion,
/// the process's CPU time over it, the encoding thread's time to encode and commit all, the
/// frames over 16.7 ms and the 99th percentile, and the side's work counts per frame.
pub fn compositing<G: Gpu>(g: &G, s: &Scene<G>, sizes: &Sizes) {
    let heap = g.tracked_heap(COMPOSITING_NEW * COMPOSITING_SLOT);
    let layers = [
        g.texture(Format::Bgra8, COMPOSITING_SIDE, COMPOSITING_SIDE, true),
        g.texture(Format::Bgra8, COMPOSITING_SIDE, COMPOSITING_SIDE, true),
    ];
    let screen = g.texture(Format::Bgra8, COMPOSITING_SIDE, COMPOSITING_SIDE, true);
    let period = Duration::from_secs_f64(1.0 / 60.0);
    let (mut wall, mut cpu, mut encode) = (vec![], vec![], vec![]);
    let mut counts = Vec::new();
    let mut next = Instant::now();
    for frame in 0..COMPOSITING_WARM + sizes.frames {
        if frame == COMPOSITING_WARM {
            counts = g.work_counts();
        }
        g.pool(|| {
            let cpu0 = process_cpu();
            let t0 = Instant::now();
            // Each new texture is made just before the command buffer that first draws into it.
            let mut news = Vec::with_capacity(COMPOSITING_NEW);
            let mut buffers = Vec::with_capacity(COMPOSITING_BUFFERS);
            for c in 0..COMPOSITING_BUFFERS {
                if c < COMPOSITING_NEW {
                    let at = c * COMPOSITING_SLOT;
                    news.push(g.heap_render_texture(
                        &heap,
                        Format::Bgra8,
                        COMPOSITING_NEW_SIDE,
                        at,
                    ));
                }
                let command_buffer = g.command_buffer();
                compositing_buffer(g, s, &command_buffer, (frame, c), &news, (&layers, &screen));
                g.commit(&command_buffer);
                buffers.push(command_buffer);
            }
            let t1 = Instant::now();
            for command_buffer in &buffers {
                g.wait(command_buffer);
            }
            let t2 = Instant::now();
            if frame >= COMPOSITING_WARM {
                wall.push(ms(t2 - t0));
                encode.push(ms(t1 - t0));
                cpu.push(ms(process_cpu() - cpu0));
            }
        });
        next += period;
        match next.checked_duration_since(Instant::now()) {
            Some(left) => std::thread::sleep(left),
            None => next = Instant::now(),
        }
    }
    let over = wall.iter().filter(|&&ms| ms > 1e3 / 60.0).count();
    emit("compositing", "over_16_7ms", "frames", &mut [over as f64]);
    let mut sorted = wall.clone();
    sorted.sort_by(f64::total_cmp);
    let p99 = sorted[((sorted.len() - 1) as f64 * 0.99).round() as usize];
    emit("compositing", "wall_p99", "ms", &mut [p99]);
    emit("compositing", "wall", "ms", &mut wall);
    emit("compositing", "encode_and_commit", "ms", &mut encode);
    emit("compositing", "process_cpu", "ms", &mut cpu);
    let after = g.work_counts();
    for ((name, before), (_, now)) in counts.iter().zip(&after) {
        let per_frame = (now - before) / sizes.frames as f64;
        emit(
            "compositing",
            &format!("rt_{name}"),
            "per frame",
            &mut [per_frame],
        );
    }
}

/// Command buffer `c` of frame `frame`: the first ten each clear one of the frame's new
/// textures (made just before it) and draw two textured quads into it; the next eight draw eight blended quads each
/// into one of the two layers, from four of the new textures and the other layer (which the
/// command buffer before drew into); one dispatches 15 small kernels and fills a buffer; the
/// last draws both layers and four new textures onto the screen.
fn compositing_buffer<G: Gpu>(
    g: &G,
    s: &Scene<G>,
    command_buffer: &G::CommandBuffer,
    (frame, c): (usize, usize),
    news: &[G::Texture],
    (layers, screen): (&[G::Texture; 2], &G::Texture),
) {
    let quad = |encoder: &mut G::RenderEncoder, n: usize| {
        let rects = &s.rects[n % RECT_BUFFERS];
        g.set_vertex_buffer(encoder, rects, (n % RECTS) * RECT_STRIDE, 0);
        g.draw_strip(encoder, 4);
    };
    match c {
        0..COMPOSITING_NEW => {
            let mut encoder = g.render_encoder(command_buffer, &news[c], true);
            g.set_render_pipeline(&mut encoder, &s.textured[0]);
            g.set_fragment_sampler(&mut encoder, &s.sampler, 0);
            g.set_fragment_texture(&mut encoder, &s.textures[(frame + c) % TEXTURES], 0);
            g.set_vertex_buffer(&mut encoder, &s.full, 0, 0);
            g.draw_strip(&mut encoder, 4);
            quad(&mut encoder, frame + c);
            g.end_render(encoder);
        }
        10..18 => {
            let (layer, other) = (&layers[c % 2], &layers[(c + 1) % 2]);
            let mut encoder = g.render_encoder(command_buffer, layer, c < 12);
            g.set_render_pipeline(&mut encoder, &s.textured[1]);
            g.set_fragment_sampler(&mut encoder, &s.sampler, 0);
            for d in 0..8 {
                let texture = match d {
                    0..4 => &news[(c + d) % COMPOSITING_NEW],
                    _ => other,
                };
                g.set_fragment_texture(&mut encoder, texture, 0);
                quad(&mut encoder, c * 8 + d);
            }
            g.end_render(encoder);
        }
        18 => {
            let mut encoder = g.compute_encoder(command_buffer);
            g.set_compute_pipeline(&mut encoder, &s.tiny[0]);
            g.set_buffer(&mut encoder, &s.tiny_values, 0, 1);
            for d in 0..15 {
                g.set_buffer(&mut encoder, &s.tiny_out, d * RECT_STRIDE, 0);
                g.dispatch(&mut encoder, 1, 64);
            }
            g.end_compute(encoder);
            g.fill(command_buffer, &s.fill, 4096);
        }
        _ => {
            let mut encoder = g.render_encoder(command_buffer, screen, frame == 0);
            g.set_render_pipeline(&mut encoder, &s.textured[1]);
            g.set_fragment_sampler(&mut encoder, &s.sampler, 0);
            for layer in layers {
                g.set_fragment_texture(&mut encoder, layer, 0);
                g.set_vertex_buffer(&mut encoder, &s.full, 0, 0);
                g.draw_strip(&mut encoder, 4);
            }
            for (n, texture) in news.iter().take(4).enumerate() {
                g.set_fragment_texture(&mut encoder, texture, 0);
                quad(&mut encoder, n * 3);
            }
            g.end_render(encoder);
        }
    }
}

/// The burst workload's period and its burst, in frames at 60 Hz: each period's last
/// `BURST_FRAMES` frames are a burst.
const BURST_PERIOD: usize = 180;
const BURST_FRAMES: usize = 60;
/// Periods run before the timed ones, and the timed ones.
const BURST_WARM: usize = 1;
const BURST_TIMED: usize = 3;
/// Command buffers a frame, calm and in a burst.
const BURST_CALM_BUFFERS: usize = 20;
const BURST_BUFFERS: usize = 60;
/// The render targets every frame makes, of the same sizes each frame ([`burst_stable_side`]).
const BURST_STABLE: usize = 15;
/// The render targets a burst's frame makes besides, each of a width no other of the last
/// `BURST_ONE_OFF_WIDTHS` made had: makings seen once.
const BURST_ONE_OFF: usize = 45;
const BURST_ONE_OFF_WIDTHS: usize = 4096;
/// The heap textures each frame makes; a burst's take sides cycling through `BURST_HEAP_SIDES`.
const BURST_HEAP: usize = 10;
const BURST_HEAP_SIDES: usize = 128;

/// Stable render target `i`'s width (each is 96 texels high).
fn burst_stable_side(i: usize) -> usize {
    96 + 32 * i
}

/// What one frame of the burst workload measured: its start, its encoding and commits, and the
/// process's CPU time at its start.
struct BurstFrame {
    start: Instant,
    encoded: Duration,
    cpu: Duration,
}

/// Frames of a window server's shape paced at 60 Hz, each period of 3 s ending in a burst of
/// 1 s, as a guest's window server bursts (a VM's work counts: about three times the command
/// buffers, many new render targets of sizes made once, heap textures of more sizes, command
/// buffers left in flight). Calm, a frame makes the `BURST_STABLE` render targets and the
/// `BURST_HEAP` heap textures, clears each and draws into it, draws the heap textures onto a
/// screen texture in its last command buffer, and waits for its `BURST_CALM_BUFFERS` command
/// buffers. In a burst it also makes `BURST_ONE_OFF` render targets of one-off widths, its heap
/// textures take sides cycling through `BURST_HEAP_SIDES`, it spreads the work over
/// `BURST_BUFFERS` command buffers, and it waits at its start for the frame before's instead of
/// its own. Calm and burst frames apart: the interval from a frame's start to the next one's (its
/// median, its 99th percentile, and those over 16.7 and 25 ms), the encoding thread's time to
/// encode and commit, and, over the same interval (a burst's command buffers complete, and their
/// textures go, while the frame after waits for its start), the process's CPU time and the side's
/// work counts.
pub fn burst<G: Gpu>(g: &G, s: &Scene<G>) {
    // Two sets of heap slots: a burst's frame leaves the frame before's in flight.
    let heap = g.tracked_heap(2 * BURST_HEAP * COMPOSITING_SLOT);
    let screen = g.texture(Format::Bgra8, COMPOSITING_SIDE, COMPOSITING_SIDE, true);
    let period = Duration::from_secs_f64(1.0 / 60.0);
    let warm = BURST_WARM * BURST_PERIOD;
    let mut frames: Vec<(bool, BurstFrame)> = Vec::new();
    // The work counts' changes over calm frames and over burst frames, each from a frame's start
    // to the next one's, and the frames of each.
    let mut counts: [(usize, Vec<(&'static str, f64)>); 2] = [(0, vec![]), (0, vec![])];
    let mut add =
        |bursting: bool, before: &[(&'static str, f64)], after: &[(&'static str, f64)]| {
            let (n, sum) = &mut counts[usize::from(bursting)];
            *n += 1;
            if sum.is_empty() {
                *sum = after.iter().map(|&(name, _)| (name, 0.0)).collect();
            }
            for ((_, total), ((_, b), (_, a))) in sum.iter_mut().zip(before.iter().zip(after)) {
                *total += a - b;
            }
        };
    let mut one_off = 0;
    let mut in_flight: Vec<G::CommandBuffer> = Vec::new();
    // The frame before's class and its start's counts, once timed.
    let mut last: Option<(bool, Vec<(&'static str, f64)>)> = None;
    let mut next = Instant::now();
    for frame in 0..warm + BURST_TIMED * BURST_PERIOD {
        let bursting = frame % BURST_PERIOD >= BURST_PERIOD - BURST_FRAMES;
        let before = g.work_counts();
        if let Some((was_bursting, then)) = last.take() {
            add(was_bursting, &then, &before);
        }
        let measured = g.pool(|| {
            let cpu = process_cpu();
            let start = Instant::now();
            for command_buffer in in_flight.drain(..) {
                g.wait(&command_buffer);
            }
            let slots = (frame % 2) * BURST_HEAP;
            let heaped: Vec<G::Texture> = (0..BURST_HEAP)
                .map(|i| {
                    let side = match bursting {
                        true => 128 + 2 * ((frame * BURST_HEAP + i) % BURST_HEAP_SIDES),
                        false => COMPOSITING_NEW_SIDE,
                    };
                    let at = (slots + i) * COMPOSITING_SLOT;
                    g.heap_render_texture(&heap, Format::Bgra8, side, at)
                })
                .collect();
            let mut targets: Vec<G::Texture> = (0..BURST_STABLE)
                .map(|i| g.texture(Format::Bgra8, burst_stable_side(i), 96, true))
                .collect();
            if bursting {
                for _ in 0..BURST_ONE_OFF {
                    let width = 128 + one_off % BURST_ONE_OFF_WIDTHS;
                    one_off += 1;
                    targets.push(g.texture(Format::Bgra8, width, 64, true));
                }
            }
            let buffers = match bursting {
                true => BURST_BUFFERS,
                false => BURST_CALM_BUFFERS,
            };
            // Each texture is cleared and drawn into by one of the command buffers before the
            // last, in turn; the last draws the heap textures onto the screen.
            let drawn: Vec<&G::Texture> = heaped.iter().chain(&targets).collect();
            let mut committed = Vec::with_capacity(buffers);
            for c in 0..buffers {
                let command_buffer = g.command_buffer();
                if c + 1 < buffers {
                    let mine = drawn
                        .iter()
                        .enumerate()
                        .filter(|(n, _)| n % (buffers - 1) == c);
                    for (n, target) in mine {
                        let mut encoder = g.render_encoder(&command_buffer, target, true);
                        g.set_render_pipeline(&mut encoder, &s.textured[0]);
                        g.set_fragment_sampler(&mut encoder, &s.sampler, 0);
                        g.set_fragment_texture(&mut encoder, &s.textures[n % TEXTURES], 0);
                        g.set_vertex_buffer(&mut encoder, &s.full, 0, 0);
                        g.draw_strip(&mut encoder, 4);
                        g.end_render(encoder);
                    }
                } else {
                    let mut encoder = g.render_encoder(&command_buffer, &screen, false);
                    g.set_render_pipeline(&mut encoder, &s.textured[1]);
                    g.set_fragment_sampler(&mut encoder, &s.sampler, 0);
                    for (n, texture) in heaped.iter().enumerate() {
                        g.set_fragment_texture(&mut encoder, texture, 0);
                        let rects = &s.rects[n % RECT_BUFFERS];
                        g.set_vertex_buffer(&mut encoder, rects, (n % RECTS) * RECT_STRIDE, 0);
                        g.draw_strip(&mut encoder, 4);
                    }
                    g.end_render(encoder);
                }
                g.commit(&command_buffer);
                committed.push(command_buffer);
            }
            let encoded = start.elapsed();
            if bursting {
                in_flight = committed;
            } else {
                for command_buffer in &committed {
                    g.wait(command_buffer);
                }
            }
            BurstFrame {
                start,
                encoded,
                cpu,
            }
        });
        if frame >= warm {
            last = Some((bursting, before));
            frames.push((bursting, measured));
        }
        next += period;
        match next.checked_duration_since(Instant::now()) {
            Some(left) => std::thread::sleep(left),
            None => next = Instant::now(),
        }
    }
    for command_buffer in in_flight.drain(..) {
        g.wait(&command_buffer);
    }
    if let Some((was_bursting, then)) = last.take() {
        add(was_bursting, &then, &g.work_counts());
    }
    for (bursting, name) in [(false, "calm"), (true, "burst")] {
        // A frame's interval: from its start to the next frame's (the run's last has none).
        let pairs = || (frames.windows(2)).filter(|pair| pair[0].0 == bursting);
        let mut intervals: Vec<f64> = pairs()
            .map(|pair| ms(pair[1].1.start - pair[0].1.start))
            .collect();
        let over = |limit: f64| intervals.iter().filter(|&&ms| ms > limit).count() as f64;
        let (over_16_7, over_25) = (over(1e3 / 60.0), over(25.0));
        emit(
            "burst",
            &format!("{name}_over_16_7ms"),
            "frames",
            &mut [over_16_7],
        );
        emit(
            "burst",
            &format!("{name}_over_25ms"),
            "frames",
            &mut [over_25],
        );
        let mut sorted = intervals.clone();
        sorted.sort_by(f64::total_cmp);
        let p99 = sorted[((sorted.len() - 1) as f64 * 0.99).round() as usize];
        emit("burst", &format!("{name}_interval_p99"), "ms", &mut [p99]);
        emit("burst", &format!("{name}_interval"), "ms", &mut intervals);
        let class = || {
            frames
                .iter()
                .filter(|(b, _)| *b == bursting)
                .map(|(_, f)| f)
        };
        let mut encoded: Vec<f64> = class().map(|f| ms(f.encoded)).collect();
        emit(
            "burst",
            &format!("{name}_encode_and_commit"),
            "ms",
            &mut encoded,
        );
        let mut cpu: Vec<f64> = pairs()
            .map(|pair| ms(pair[1].1.cpu - pair[0].1.cpu))
            .collect();
        emit("burst", &format!("{name}_process_cpu"), "ms", &mut cpu);
        let (n, sum) = &counts[usize::from(bursting)];
        for (count, total) in sum {
            let per_frame = total / *n as f64;
            emit(
                "burst",
                &format!("{name}_rt_{count}"),
                "per frame",
                &mut [per_frame],
            );
        }
    }
}

/// The icb workload's command counts per execution.
const ICB_COMMANDS: [usize; 4] = [16, 64, 256, 4096];
/// The words of one command's window of the icb workload's data.
const ICB_WINDOW: usize = 64;
/// Untimed frames before the icb workload's timed ones.
const ICB_WARM: usize = 3;
/// The icb workload's live states made with `supportIndirectCommandBuffers`: the commands name
/// the first four at most, the rest stay alive unnamed, as an app's other such states do. Each
/// side of the runtime's bounds on the states an execution runs with on the GPU alone, at each
/// count of [`ICB_COMMANDS`] (a concurrent encoder's: 20, 8, 5 and 4; a serial encoder's: 44, 20,
/// 14 and 8).
const ICB_STATES: [usize; 10] = [1, 4, 5, 6, 8, 9, 14, 15, 20, 21];
/// The most states the icb workload's commands name (`icb_step0` to `icb_step3`).
const ICB_NAMED: usize = 4;

/// The icb workload (main.rs): per frame, one command buffer of a compute encoder whose kernel
/// (`icb_encode`) encodes `commands` compute commands into an indirect command buffer, one a
/// thread, and a second compute encoder that executes them, each command one threadgroup of 64
/// threads of one of the first [`ICB_NAMED`] at most of `states` states made with
/// `supportIndirectCommandBuffers` (the only such states alive), in turn; committed and waited
/// for. For a serial and a concurrent executing encoder, each count of [`ICB_STATES`] and of
/// [`ICB_COMMANDS`]: the frame's wall time (the command buffer's creation to the wait's return),
/// the encoding thread's (through the commit's return) and the process's CPU time, and on the
/// runtime's side its work counts per frame (`rt_*`). The first frame's output is checked: each
/// command's window holds its state's step of zero.
pub fn icb<G: Gpu>(g: &G, library: &G::Library, sizes: &Sizes) {
    let encode = g.compute_pipeline(library, "icb_encode");
    for states in ICB_STATES {
        let named = states.min(ICB_NAMED);
        let steps: Vec<G::ComputePipeline> = (0..states)
            .map(|k| g.indirect_compute_pipeline(library, &format!("icb_step{}", k % ICB_NAMED)))
            .collect();
        for (concurrent, commands) in [false, true]
            .into_iter()
            .flat_map(|c| ICB_COMMANDS.map(|n| (c, n)))
        {
            let icb = g.indirect_commands(commands);
            let mut arguments = g.commands_id(&icb).to_le_bytes().to_vec();
            for k in 0..ICB_NAMED {
                arguments.extend(g.pipeline_id(&steps[k.min(named - 1)]).to_le_bytes());
            }
            let arguments = g.buffer_with(&arguments);
            let count = g.buffer_with(&(named as u32).to_le_bytes());
            let data = g.buffer_with(&vec![0u8; commands * ICB_WINDOW * 4]);
            let threads = commands.min(64);
            let (mut wall, mut encoding, mut cpu) = (vec![], vec![], vec![]);
            let mut counts = Vec::new();
            for frame in 0..ICB_WARM + sizes.timed {
                if frame == ICB_WARM {
                    counts = g.work_counts();
                }
                g.pool(|| {
                    let cpu0 = process_cpu();
                    let t0 = Instant::now();
                    let command_buffer = g.command_buffer();
                    let mut encoder = g.compute_encoder(&command_buffer);
                    g.set_compute_pipeline(&mut encoder, &encode);
                    g.set_buffer(&mut encoder, &arguments, 0, 0);
                    g.set_buffer(&mut encoder, &data, 0, 1);
                    g.set_buffer(&mut encoder, &count, 0, 2);
                    g.use_commands(&mut encoder, &icb);
                    g.dispatch(&mut encoder, commands / threads, threads);
                    g.end_compute(encoder);
                    let mut encoder = match concurrent {
                        true => g.concurrent_compute_encoder(&command_buffer),
                        false => g.compute_encoder(&command_buffer),
                    };
                    g.use_buffer(&mut encoder, &data);
                    g.execute_commands(&mut encoder, &icb, commands);
                    g.end_compute(encoder);
                    g.commit(&command_buffer);
                    let t1 = Instant::now();
                    g.wait(&command_buffer);
                    let t2 = Instant::now();
                    if frame >= ICB_WARM {
                        wall.push(ms(t2 - t0));
                        encoding.push(ms(t1 - t0));
                        cpu.push(ms(process_cpu() - cpu0));
                    }
                });
                if frame == 0 {
                    let mut bytes = vec![0u8; commands * ICB_WINDOW * 4];
                    g.read_buffer(&data, &mut bytes);
                    for (word, value) in bytes.as_chunks::<4>().0.iter().enumerate() {
                        let expected = (word / ICB_WINDOW % named) as u32 + 1;
                        assert_eq!(
                            u32::from_le_bytes(*value),
                            expected,
                            "icb: word {word} of {commands} commands of {states} states"
                        );
                    }
                }
            }
            let kind = if concurrent { "concurrent" } else { "serial" };
            let at = format!("{kind}_s{states}_c{commands}");
            emit("icb", &format!("frame_{at}"), "ms", &mut wall);
            emit("icb", &format!("encode_{at}"), "ms", &mut encoding);
            emit("icb", &format!("process_cpu_{at}"), "ms", &mut cpu);
            let after = g.work_counts();
            for ((name, before), (_, now)) in counts.iter().zip(&after) {
                if [
                    "queue_submissions",
                    "pipeline_barriers",
                    "dispatches",
                    "dispatches_indirect",
                ]
                .contains(name)
                {
                    let per_frame = (now - before) / sizes.timed as f64;
                    emit(
                        "icb",
                        &format!("rt_{name}_{at}"),
                        "per frame",
                        &mut [per_frame],
                    );
                }
            }
        }
    }
}

/// The commands of each of the icb_cpu workload's buffers.
const ICB_CPU_COMMANDS: usize = 4096;

/// The icb_cpu workload (main.rs): the CPU's encoding of indirect commands and their read-back,
/// each call's time. Per pass, over [`ICB_CPU_COMMANDS`] commands of a buffer of dispatches and
/// one of draws (`Shared`, inheriting nothing), each setter on every command in turn, then each
/// getter; each call takes the command (`indirectComputeCommandAtIndex:` or
/// `indirectRenderCommandAtIndex:`) first, as a program that encodes once per command does. The
/// medians over the passes, in nanoseconds a call; [`ICB_WARM`] untimed passes first.
pub fn icb_cpu<G: Gpu>(g: &G, library: &G::Library, sizes: &Sizes) {
    let state = g.indirect_compute_pipeline(library, "icb_step0");
    let draw = g.indirect_render_pipeline(library, "icb_draw_vertex", "icb_draw_fragment");
    let buffer = g.buffer(ICB_CPU_COMMANDS * 256);
    let dispatches = g.cpu_indirect_commands(ICB_CPU_COMMANDS);
    let draws = g.cpu_indirect_draws(ICB_CPU_COMMANDS);
    let names = [
        "set_compute_pipeline",
        "set_kernel_buffer",
        "dispatch_threadgroups",
        "set_barrier",
        "kernel_buffer_address",
        "has_barrier",
        "set_render_pipeline",
        "set_vertex_buffer",
        "draw_primitives",
        "vertex_buffer_address",
    ];
    let mut times: Vec<Vec<f64>> = vec![Vec::new(); names.len()];
    let mut sink = 0u64;
    for pass in 0..ICB_WARM + sizes.timed {
        g.pool(|| {
            let mut timed = |at: usize, call: &mut dyn FnMut(usize)| {
                let t = Instant::now();
                for index in 0..ICB_CPU_COMMANDS {
                    call(index);
                }
                if pass >= ICB_WARM {
                    let each = t.elapsed().as_secs_f64() * 1e9 / ICB_CPU_COMMANDS as f64;
                    times[at].push(each);
                }
            };
            timed(0, &mut |i| g.command_pipeline(&dispatches, i, &state));
            timed(1, &mut |i| {
                g.command_kernel_buffer(&dispatches, i, &buffer, i * 256)
            });
            timed(2, &mut |i| g.command_dispatch(&dispatches, i, 1 + i % 4));
            timed(3, &mut |i| g.command_barrier(&dispatches, i));
            timed(4, &mut |i| {
                sink = sink.wrapping_add(g.command_kernel_buffer_address(&dispatches, i))
            });
            timed(5, &mut |i| {
                sink = sink.wrapping_add(u64::from(g.command_has_barrier(&dispatches, i)))
            });
            timed(6, &mut |i| g.draw_command_pipeline(&draws, i, &draw));
            timed(7, &mut |i| {
                g.draw_command_vertex_buffer(&draws, i, &buffer, i * 256)
            });
            timed(8, &mut |i| g.draw_command_draw(&draws, i));
            timed(9, &mut |i| {
                sink = sink.wrapping_add(g.draw_command_vertex_buffer_address(&draws, i))
            });
        });
    }
    std::hint::black_box(sink);
    for (name, times) in names.iter().zip(&mut times) {
        emit("icb_cpu", name, "ns", times);
    }
}

/// The icb_draws workload's objects, as many as Apple's "Encoding Indirect Command Buffers on the GPU"
/// draws (`AAPLNumObjects`), on a grid of `ICB_DRAW_GRID` columns.
const ICB_DRAW_OBJECTS: usize = 65536;
const ICB_DRAW_GRID: usize = 256;
/// Each object's mesh: a hexagon's six triangles.
const ICB_DRAW_VERTICES: usize = 18;

/// The icb_draws workload's objects and their indirect command buffer (`main.rs`'s module docs).
struct IcbDraws<G: Gpu> {
    encode: G::ComputePipeline,
    draw: G::RenderPipeline,
    /// The frame state, the objects, the mesh, the argument buffer holding the buffer's ID and the
    /// draws' primitive type (`icb_draw_encode_any_type`'s).
    frame: G::Buffer,
    objects: G::Buffer,
    mesh: G::Buffer,
    container: G::Buffer,
    primitive: G::Buffer,
    commands: G::IndirectCommands,
    target: G::Texture,
}

impl<G: Gpu> IcbDraws<G> {
    /// The objects, encoded by `kernel`.
    fn new(g: &G, library: &G::Library, kernel: &str) -> IcbDraws<G> {
        // The grid spans a little more than the view, so its edge objects are culled.
        let span = 2.2f32;
        let cell = span / ICB_DRAW_GRID as f32;
        let mut objects = Vec::with_capacity(ICB_DRAW_OBJECTS * 16);
        for object in 0..ICB_DRAW_OBJECTS {
            let (column, row) = (object % ICB_DRAW_GRID, object / ICB_DRAW_GRID);
            let x = -span / 2.0 + (column as f32 + 0.5) * cell;
            let y = -span / 2.0 + (row as f32 + 0.5) * cell;
            for word in [x.to_bits(), y.to_bits(), (cell * 0.4).to_bits()] {
                objects.extend_from_slice(&word.to_le_bytes());
            }
            objects.extend_from_slice(&(ICB_DRAW_VERTICES as u32).to_le_bytes());
        }
        let mut mesh = Vec::with_capacity(ICB_DRAW_VERTICES * 8);
        let corner = |k: usize| {
            let angle = k as f32 * std::f32::consts::TAU / 6.0;
            [angle.cos() * cell * 0.4, angle.sin() * cell * 0.4]
        };
        for k in 0..6 {
            for [x, y] in [[0.0, 0.0], corner(k), corner(k + 1)] {
                mesh.extend_from_slice(&f32::to_le_bytes(x));
                mesh.extend_from_slice(&f32::to_le_bytes(y));
            }
        }
        let frame: Vec<u8> = [0.0f32, 0.0, 1.0, 1.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let commands = g.indirect_draws(ICB_DRAW_OBJECTS, 3);
        IcbDraws {
            encode: g.compute_pipeline(library, kernel),
            draw: g.indirect_render_pipeline(library, "icb_draw_vertex", "icb_draw_fragment"),
            frame: g.buffer_with(&frame),
            objects: g.buffer_with(&objects),
            mesh: g.buffer_with(&mesh),
            container: g.buffer_with(&g.commands_id(&commands).to_le_bytes()),
            // `MTLPrimitiveTypeTriangle`.
            primitive: g.buffer_with(&3u32.to_le_bytes()),
            commands,
            target: g.texture(Format::Bgra8, 1920, 1080, true),
        }
    }

    /// One frame, as the sample's: a blit resets the commands, a kernel culls the objects and
    /// encodes a command for each visible one, and a render pass executes them all.
    fn encode(&self, g: &G, command_buffer: &G::CommandBuffer) {
        g.reset_commands(command_buffer, &self.commands, ICB_DRAW_OBJECTS);
        let mut encoder = g.compute_encoder(command_buffer);
        g.set_compute_pipeline(&mut encoder, &self.encode);
        g.set_buffer(&mut encoder, &self.frame, 0, 0);
        g.set_buffer(&mut encoder, &self.objects, 0, 1);
        g.set_buffer(&mut encoder, &self.mesh, 0, 2);
        g.set_buffer(&mut encoder, &self.container, 0, 3);
        g.set_buffer(&mut encoder, &self.primitive, 0, 4);
        g.use_commands(&mut encoder, &self.commands);
        g.dispatch(&mut encoder, ICB_DRAW_OBJECTS / 64, 64);
        g.end_compute(encoder);
        let mut encoder = g.render_encoder(command_buffer, &self.target, true);
        g.set_render_pipeline(&mut encoder, &self.draw);
        for buffer in [&self.frame, &self.objects, &self.mesh] {
            g.use_in_vertex(&mut encoder, buffer);
        }
        g.execute_draws(&mut encoder, &self.commands, ICB_DRAW_OBJECTS);
        g.end_render(encoder);
    }
}

/// The icb_draws workload (module docs of `main.rs`): per frame, the encoding thread's wall time
/// through `endEncoding`, the commit call's, the commit's return to the wait's, the whole frame's,
/// the process's CPU time, the encoding thread's CPU time through the commit's return, and the GPU
/// time where the side measures it; the same of commands whose primitive type the encoding kernel
/// reads from a buffer (`icb_draws_any_type`), and on the runtime's side its counts of commands
/// drawn on the GPU alone and records read on the host, per frame.
pub fn icb_draws<G: Gpu>(g: &G, library: &G::Library, sizes: &Sizes) {
    for (name, kernel) in [
        ("icb_draws", "icb_draw_encode"),
        ("icb_draws_any_type", "icb_draw_encode_any_type"),
    ] {
        let scene = IcbDraws::new(g, library, kernel);
        let mut times: [Vec<f64>; 7] = Default::default();
        let mut counts = Vec::new();
        for frame in 0..sizes.timed + 3 {
            if frame == 3 {
                counts = g.work_counts();
            }
            g.pool(|| {
                let cpu = process_cpu();
                let thread = thread_cpu();
                let start = Instant::now();
                let command_buffer = g.command_buffer();
                scene.encode(g, &command_buffer);
                let encoded = Instant::now();
                g.commit(&command_buffer);
                let committed = Instant::now();
                let thread = thread_cpu() - thread;
                g.wait(&command_buffer);
                let done = Instant::now();
                let cpu = process_cpu() - cpu;
                if frame < 3 {
                    return;
                }
                times[6].push(ms(thread));
                times[0].push(ms(encoded - start));
                times[1].push(ms(committed - encoded));
                times[2].push(ms(done - committed));
                times[3].push(ms(done - start));
                times[4].push(ms(cpu));
                if let Some(gpu) = g.gpu_seconds(&command_buffer) {
                    times[5].push(gpu * 1e3);
                }
            });
        }
        let [encode, commit, wait, frame, cpu, gpu, thread] = &mut times;
        emit(name, "encode", "ms", encode);
        emit(name, "commit", "ms", commit);
        emit(name, "commit_to_done", "ms", wait);
        emit(name, "frame", "ms", frame);
        emit(name, "cpu", "ms", cpu);
        emit(name, "thread_cpu", "ms", thread);
        if !gpu.is_empty() {
            emit(name, "gpu", "ms", gpu);
        }
        let after = g.work_counts();
        for ((counted, before), (_, now)) in counts.iter().zip(&after) {
            if matches!(
                *counted,
                "gpu_form_commands"
                    | "merged_draw_commands"
                    | "indirect_records_read"
                    | "draws"
                    | "pipeline_barriers"
            ) {
                let per_frame = (now - before) / sizes.timed as f64;
                emit(name, &format!("rt_{counted}"), "count", &mut [per_frame]);
            }
        }
    }
}

/// A frame of the calls workload, as a guest cuts one (module docs of `main.rs`): command
/// buffers, each of a render pass of `draws` draws, two compute encoders of `dispatches`
/// dispatches and a blit encoder of `blits` copies.
#[derive(Clone, Copy)]
struct CallShape {
    name: &'static str,
    command_buffers: usize,
    draws: usize,
    dispatches: usize,
    blits: usize,
}

/// One command buffer of long encoders, and many of short ones.
const CALL_SHAPES: [CallShape; 2] = [
    CallShape {
        name: "wide",
        command_buffers: 1,
        draws: 512,
        dispatches: 64,
        blits: 32,
    },
    CallShape {
        name: "many",
        command_buffers: 64,
        draws: 16,
        dispatches: 1,
        blits: 1,
    },
];

/// The calls a frame of the calls workload times, each a step.
const CALL_STEPS: [&str; 16] = [
    "command_buffer",
    "render_encoder",
    "set_state",
    "set_vertex_buffers",
    "set_fragment_texture",
    "set_fragment_sampler",
    "set_fragment_buffer",
    "draw",
    "end_render",
    "compute_encoder",
    "dispatch",
    "dispatch_resident",
    "end_compute",
    "blit_copy",
    "end_blit",
    "handler_commit",
];

/// The buffers the second compute encoder of each command buffer makes resident.
const CALL_RESIDENT: usize = 16;

/// Each step's total and count over a frame: nanoseconds, or with `allocs` the allocations
/// (`crate::allocs`) on the encoding thread.
#[derive(Clone, Copy, Default)]
struct StepTally([(u64, u64); CALL_STEPS.len()]);

impl StepTally {
    fn time<T>(&mut self, step: usize, allocs: bool, call: impl FnOnce() -> T) -> T {
        let (out, amount) = match allocs {
            true => {
                let before = crate::allocs::this_thread();
                let out = call();
                (out, crate::allocs::this_thread() - before)
            }
            false => {
                let start = Instant::now();
                let out = call();
                (out, start.elapsed().as_nanos() as u64)
            }
        };
        let (total, count) = &mut self.0[step];
        *total += amount;
        *count += 1;
        out
    }
}

/// The calls workload's objects.
struct CallObjects<G: Gpu> {
    pass: G::Pass,
    positions: G::Buffer,
    colours: G::Buffer,
    uniforms: G::Buffer,
    states: [G::RenderPipeline; 2],
    input: G::Buffer,
    output: G::Buffer,
    resident: Vec<G::Buffer>,
    from: G::Buffer,
    to: G::Buffer,
}

impl<G: Gpu> CallObjects<G> {
    /// One command buffer of `shape`, its calls timed (or their allocations counted) into `t`,
    /// committed with a completed handler that sends on its channel.
    fn command_buffer(
        &self,
        g: &G,
        s: &Scene<G>,
        shape: CallShape,
        (t, allocs): (&mut StepTally, bool),
    ) -> (G::CommandBuffer, mpsc::Receiver<()>) {
        let cb = t.time(0, allocs, || g.command_buffer());
        let mut e = t.time(1, allocs, || g.pass_encoder(&cb, &self.pass));
        let vertex = [&self.positions, &self.colours];
        for i in 0..shape.draws {
            let textured = (i / 8) % 2 == 0;
            if i % 8 == 0 {
                let state = &self.states[usize::from(!textured)];
                t.time(2, allocs, || g.set_render_pipeline(&mut e, state));
            }
            t.time(3, allocs, || {
                g.set_vertex_buffers(&mut e, &vertex, &[0, (i % 64) * 16], 0)
            });
            if textured {
                t.time(4, allocs, || {
                    g.set_fragment_texture(&mut e, &s.textures[0], 0)
                });
                t.time(5, allocs, || g.set_fragment_sampler(&mut e, &s.sampler, 0));
            } else {
                t.time(6, allocs, || {
                    g.set_fragment_buffer(&mut e, &self.uniforms, (i % 256) * 256, 0)
                });
            }
            t.time(7, allocs, || g.draw_strip(&mut e, 4));
        }
        t.time(8, allocs, || g.end_render(e));
        let resident: Vec<&G::Buffer> = self.resident.iter().collect();
        for (step, used) in [(10, &resident[..0]), (11, &resident[..])] {
            let mut e = t.time(9, allocs, || g.compute_encoder(&cb));
            g.set_compute_pipeline(&mut e, &s.tiny[0]);
            if !used.is_empty() {
                g.use_buffers_read(&mut e, used);
            }
            for i in 0..shape.dispatches {
                let at = i * 256;
                t.time(step, allocs, || {
                    g.set_buffer(&mut e, &self.output, at, 0);
                    g.set_buffer(&mut e, &self.input, at, 1);
                    g.dispatch(&mut e, 1, 64);
                });
            }
            t.time(12, allocs, || g.end_compute(e));
        }
        let mut blit = g.blit_encoder(&cb);
        for i in 0..shape.blits {
            let at = i * 1024;
            t.time(13, allocs, || {
                g.copy_buffer(&mut blit, (&self.from, at), (&self.to, at), 1024)
            });
        }
        t.time(14, allocs, || g.end_blit(blit));
        let (tx, rx) = mpsc::channel();
        t.time(15, allocs, || {
            g.on_completed(&cb, Box::new(move || tx.send(()).unwrap()));
            g.commit(&cb);
        });
        (cb, rx)
    }

    /// One frame of `shape`: its command buffers committed, then each one's handler and wait.
    fn frame(&self, g: &G, s: &Scene<G>, shape: CallShape, allocs: bool) -> StepTally {
        g.pool(|| {
            let mut t = StepTally::default();
            let committed: Vec<_> = (0..shape.command_buffers)
                .map(|_| self.command_buffer(g, s, shape, (&mut t, allocs)))
                .collect();
            for (cb, rx) in committed {
                rx.recv().unwrap();
                g.wait(&cb);
            }
            t
        })
    }
}

/// The buffers the calls workload's `resident_*` variants make resident, of 256 bytes each.
const DRAWN_RESIDENT: usize = 4096;

/// The draws of a `resident_*` variant's render encoder.
const RESIDENT_DRAWS: usize = 512;

/// The calls workload's `resident_*` variants, each run only where named: a render encoder of the
/// buffer-fragment pipeline state, its vertex and fragment buffers set, one
/// `useResources:count:usage:stages:` (reading, vertex and fragment) of 0, 1, 256 or 4096 buffers,
/// then [`RESIDENT_DRAWS`] draws of a strip with nothing changed between them. Per variant: the
/// `useResources` call, the encoding thread's wall time per draw after it, and the command
/// buffer's whole time to its wait's return.
fn calls_resident<G: Gpu>(g: &G, objects: &CallObjects<G>, sizes: &Sizes, variants: &[String]) {
    let wanted: Vec<(&str, usize)> = [("0", 0), ("1", 1), ("256", 256), ("4096", 4096)]
        .into_iter()
        .filter(|(name, _)| variants.iter().any(|v| *v == format!("resident_{name}")))
        .collect();
    if wanted.is_empty() {
        return;
    }
    let resident: Vec<G::Buffer> = (0..DRAWN_RESIDENT).map(|_| g.buffer(256)).collect();
    let vertex = [&objects.positions, &objects.colours];
    for (name, count) in wanted {
        let list: Vec<&G::Buffer> = resident[..count].iter().collect();
        let (mut using, mut draws, mut total) = (vec![], vec![], vec![]);
        for run in 0..sizes.timed + 3 {
            g.pool(|| {
                let t0 = Instant::now();
                let cb = g.command_buffer();
                let mut e = g.pass_encoder(&cb, &objects.pass);
                g.set_render_pipeline(&mut e, &objects.states[1]);
                g.set_vertex_buffers(&mut e, &vertex, &[0, 0], 0);
                g.set_fragment_buffer(&mut e, &objects.uniforms, 0, 0);
                let t1 = Instant::now();
                g.use_buffers_drawn(&mut e, &list);
                let t2 = Instant::now();
                for _ in 0..RESIDENT_DRAWS {
                    g.draw_strip(&mut e, 4);
                }
                let t3 = Instant::now();
                g.end_render(e);
                g.commit(&cb);
                g.wait(&cb);
                let t4 = Instant::now();
                if run >= 3 {
                    using.push((t2 - t1).as_secs_f64() * 1e6);
                    draws.push((t3 - t2).as_secs_f64() * 1e9 / RESIDENT_DRAWS as f64);
                    total.push((t4 - t0).as_secs_f64() * 1e3);
                }
            });
        }
        let metric = |what: &str| format!("resident_{name}_{what}");
        emit("calls", &metric("use_call"), "us", &mut using);
        emit("calls", &metric("draw"), "ns/call", &mut draws);
        emit("calls", &metric("cb_total"), "ms", &mut total);
    }
}

/// The consumer's frames, call by call (module docs of `main.rs`): each step's mean a call over
/// each timed frame, then its allocations a call on the encoding thread over one more frame.
/// The `resident_*` variants run only where named ([`calls_resident`]).
pub fn calls<G: Gpu>(
    g: &G,
    s: &Scene<G>,
    library: &G::Library,
    sizes: &Sizes,
    variants: &[String],
) {
    let floats_of = |n: usize, f: &dyn Fn(usize) -> [f32; 4]| {
        let values: Vec<f32> = (0..n).flat_map(f).collect();
        g.buffer_with(&floats(&values))
    };
    let state =
        |fragment| g.render_pipeline(library, "calls_vertex", fragment, Format::Bgra8, false);
    let objects = CallObjects {
        pass: g.pass(&s.target),
        // A strip of four corners of a small square, then nothing.
        positions: floats_of(256, &|v| match v {
            0..4 => [
                -0.1 + 0.2 * (v & 1) as f32,
                -0.1 + 0.2 * (v >> 1) as f32,
                0.0,
                1.0,
            ],
            _ => [0.0; 4],
        }),
        colours: floats_of(256, &|v| [0.5, (v % 7) as f32 / 7.0, 0.25, 1.0]),
        uniforms: floats_of(4096, &|v| [(v % 5) as f32 / 5.0, 0.5, 0.5, 1.0]),
        states: [
            state("calls_texture_fragment"),
            state("calls_buffer_fragment"),
        ],
        input: g.buffer_with(&vec![1u8; 65536]),
        output: g.buffer(65536),
        resident: (0..CALL_RESIDENT).map(|_| g.buffer(4096)).collect(),
        from: g.buffer_with(&vec![0x3cu8; 65536]),
        to: g.buffer(65536),
    };
    const WARM: usize = 5;
    let shapes = CALL_SHAPES
        .into_iter()
        .filter(|shape| variants.is_empty() || variants.iter().any(|v| v == shape.name));
    for shape in shapes {
        let mut means: [Vec<f64>; CALL_STEPS.len()] = Default::default();
        let (mut frames, mut walls) = (vec![], vec![]);
        for frame in 0..WARM + sizes.timed {
            let start = Instant::now();
            let t = objects.frame(g, s, shape, false);
            let wall = start.elapsed();
            if frame >= WARM {
                walls.push(wall.as_secs_f64() * 1e6);
                let mut sum = 0.0;
                for (step, &(total, count)) in t.0.iter().enumerate() {
                    if count > 0 {
                        means[step].push(total as f64 / count as f64);
                    }
                    sum += total as f64;
                }
                frames.push(sum / 1e3);
            }
        }
        let name = shape.name;
        for (step, values) in CALL_STEPS.iter().zip(&mut means) {
            if !values.is_empty() {
                emit("calls", &format!("{name}_{step}"), "ns/call", values);
            }
        }
        emit("calls", &format!("{name}_frame_calls"), "us", &mut frames);
        // The frame from its first call until its last command buffer's handler ran and its wait
        // returned: what the calls cost, and how soon their work went to the GPU and completed.
        emit("calls", &format!("{name}_frame_wall"), "us", &mut walls);
        let t = crate::allocs::counting(|| objects.frame(g, s, shape, true));
        for (step, &(total, count)) in CALL_STEPS.iter().zip(&t.0) {
            if count > 0 {
                emit(
                    "calls",
                    &format!("{name}_{step}_allocs"),
                    "allocs/call",
                    &mut [total as f64 / count as f64],
                );
            }
        }
    }
    calls_resident(g, &objects, sizes, variants);
}

/// The provoking workload's draws in a frame: each two small triangles of six vertices of their own
/// in a 64 by 32 grid over the target (module docs of `main.rs`).
const FLAT_VERTICES: usize = 6;

/// How the provoking workload draws.
#[derive(Clone, Copy)]
enum FlatForm {
    Direct,
    Indexed,
    IndexedStrip,
    Indirect,
    IndexedIndirect,
}

impl FlatForm {
    fn name(self) -> &'static str {
        match self {
            FlatForm::Direct => "direct",
            FlatForm::Indexed => "indexed",
            FlatForm::IndexedStrip => "indexed_strip",
            FlatForm::Indirect => "indirect",
            FlatForm::IndexedIndirect => "indexed_indirect",
        }
    }
}

/// An OpenGL guest's flat-shaded draws (module docs of `main.rs`): per draw, at
/// `setProvokingVertexMode:` 0 and at 1 (Apple's GL's last vertex), each form's encoding time,
/// the process's CPU time, and the frame from the command buffer's creation to the wait's return.
pub fn provoking<G: Gpu>(g: &G, library: &G::Library, sizes: &Sizes) {
    let n = sizes.draws;
    assert!(
        n * FLAT_VERTICES <= 65536,
        "--draws {n}: past 16-bit indices"
    );
    let mut positions = Vec::with_capacity(n * FLAT_VERTICES * 16);
    for draw in 0..n {
        let cell = draw % 2048;
        let (x, y) = ((cell % 64) as f32, (cell / 64) as f32);
        let (x0, x1) = (x / 32.0 - 1.0, (x + 0.8) / 32.0 - 1.0);
        let (y0, y1) = (y / 16.0 - 1.0, (y + 0.8) / 16.0 - 1.0);
        for (px, py) in [(x0, y0), (x1, y0), (x0, y1), (x0, y1), (x1, y0), (x1, y1)] {
            positions.extend(floats(&[px, py, 0.0, 1.0]));
        }
    }
    let words = |w: &[u32]| -> Vec<u8> { w.iter().flat_map(|w| w.to_le_bytes()).collect() };
    let indices: Vec<u8> = (0..n * FLAT_VERTICES)
        .flat_map(|i| (i as u16).to_le_bytes())
        .collect();
    // Each draw's two triangles as a strip of four of its six vertices: the list's second
    // triangle is the strip's second, in its order.
    let strip_indices: Vec<u8> = (0..n * FLAT_VERTICES)
        .step_by(FLAT_VERTICES)
        .flat_map(|at| [at, at + 1, at + 2, at + 5])
        .flat_map(|i| (i as u16).to_le_bytes())
        .collect();
    let arguments: Vec<u32> = (0..n)
        .flat_map(|d| [FLAT_VERTICES as u32, 1, (d * FLAT_VERTICES) as u32, 0])
        .collect();
    let indexed_arguments: Vec<u32> = (0..n)
        .flat_map(|d| [FLAT_VERTICES as u32, 1, (d * FLAT_VERTICES) as u32, 0, 0])
        .collect();
    let positions = g.buffer_with(&positions);
    let indices = g.buffer_with(&indices);
    let strip_indices = g.buffer_with(&strip_indices);
    let arguments = g.buffer_with(&words(&arguments));
    let indexed_arguments = g.buffer_with(&words(&indexed_arguments));
    let pipeline = g.render_pipeline(
        library,
        "flat_vertex",
        "flat_fragment",
        Format::Bgra8,
        false,
    );
    let target = g.texture(Format::Bgra8, TARGET_SIDE, TARGET_SIDE, true);
    let forms = [
        FlatForm::Direct,
        FlatForm::Indexed,
        FlatForm::IndexedStrip,
        FlatForm::Indirect,
        FlatForm::IndexedIndirect,
    ];
    for form in forms {
        for mode in [0u64, 1] {
            let name = format!("{}_mode{mode}", form.name());
            let counts = g.work_counts();
            let mut times: [Vec<f64>; 4] = Default::default();
            for run in 0..sizes.timed + 3 {
                g.pool(|| {
                    let cpu0 = process_cpu();
                    let t0 = Instant::now();
                    let command_buffer = g.command_buffer();
                    let mut encoder = g.render_encoder(&command_buffer, &target, true);
                    g.set_render_pipeline(&mut encoder, &pipeline);
                    g.set_vertex_buffer(&mut encoder, &positions, 0, 0);
                    g.set_provoking_vertex_mode(&mut encoder, mode);
                    for draw in 0..n {
                        let at = draw * FLAT_VERTICES;
                        match form {
                            FlatForm::Direct => g.draw_triangles(&mut encoder, at, FLAT_VERTICES),
                            FlatForm::Indexed => {
                                g.draw_indexed_at(&mut encoder, FLAT_VERTICES, &indices, at * 2)
                            }
                            FlatForm::IndexedStrip => {
                                g.draw_indexed_strip_at(&mut encoder, 4, &strip_indices, draw * 8)
                            }
                            FlatForm::Indirect => {
                                g.draw_triangles_indirect(&mut encoder, &arguments, draw * 16)
                            }
                            FlatForm::IndexedIndirect => g.draw_indexed_indirect(
                                &mut encoder,
                                &indices,
                                &indexed_arguments,
                                draw * 20,
                            ),
                        }
                    }
                    g.end_render(encoder);
                    let t1 = Instant::now();
                    g.commit(&command_buffer);
                    g.wait(&command_buffer);
                    let t2 = Instant::now();
                    let cpu1 = process_cpu();
                    if run >= 3 {
                        times[0].push(us(t1 - t0) / n as f64);
                        times[1].push(us(cpu1 - cpu0) / n as f64);
                        times[2].push(ms(t2 - t1));
                        times[3].push(us(t2 - t0) / n as f64);
                    }
                });
            }
            let [encode, cpu, wait, frame] = &mut times;
            emit("provoking", &format!("{name}_encode"), "us/draw", encode);
            emit("provoking", &format!("{name}_process_cpu"), "us/draw", cpu);
            emit("provoking", &format!("{name}_commit_to_done"), "ms", wait);
            emit("provoking", &format!("{name}_frame"), "us/draw", frame);
            let after = g.work_counts();
            for ((counted, before), (_, now)) in counts.iter().zip(&after) {
                if matches!(
                    *counted,
                    "provoking_last_pipeline_draws"
                        | "provoking_listed_draws"
                        | "gathered_draws"
                        | "bounded_run_draws"
                        | "dispatches"
                        | "pipeline_barriers"
                ) {
                    let per_frame = (now - before) / (sizes.timed + 3) as f64;
                    emit(
                        "provoking",
                        &format!("{name}_rt_{counted}"),
                        "count",
                        &mut [per_frame],
                    );
                }
            }
        }
    }
}
