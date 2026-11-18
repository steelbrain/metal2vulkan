//! The authored workloads: the kernels of the public fixture `harness/workloads.metal`, their
//! inputs made here from a fixed seed, and the CPU reference of their results. Every result is
//! exact on any device but the ALU workload's, whose `float` chains are held to a tolerance.

use crate::plan::{
    seeded_words as words, Binding, BufferPlan, Constant, Dispatch, Expect, Filter, Grid, Plan,
    SamplerPlan, Scaled, Source, TextureExpect, TexturePlan, Values,
};
use crate::{Result, Storage};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use std::ops::Range;

/// An authored workload: its name, and the builder of its plan. A plan holds its inputs and
/// expected results (up to 128 MiB), so each is built when it runs.
#[derive(Clone, Copy, Debug)]
pub struct Workload {
    /// Its name in reports and in `--only`.
    pub name: &'static str,
    /// Its plan's builder.
    pub plan: fn() -> Plan,
}

/// The authored workloads.
pub const WORKLOADS: [Workload; 38] = [
    Workload {
        name: "bandwidth",
        plan: bandwidth,
    },
    Workload {
        name: "alu",
        plan: alu,
    },
    Workload {
        name: "threadgroup_scan",
        plan: threadgroup_scan,
    },
    Workload {
        name: "simd_reduce",
        plan: simd_reduce,
    },
    Workload {
        name: "image_filter",
        plan: image_filter,
    },
    Workload {
        name: "dispatch_chain",
        plan: dispatch_chain,
    },
    Workload {
        name: "row_cells",
        plan: row_cells,
    },
    Workload {
        name: "frame",
        plan: frame,
    },
    Workload {
        name: "frame_5cb",
        plan: frame_5cb,
    },
    Workload {
        name: "long_4cb",
        plan: long_4cb,
    },
    Workload {
        name: "burst_160",
        plan: burst_160,
    },
    Workload {
        name: "burst_160_shared",
        plan: burst_160_shared,
    },
    Workload {
        name: "burst_2000",
        plan: burst_2000,
    },
    Workload {
        name: "burst_2000x5",
        plan: burst_2000x5,
    },
    Workload {
        name: "narrow_ints",
        plan: narrow_ints,
    },
    Workload {
        name: "zero_products",
        plan: zero_products,
    },
    Workload {
        name: "sampling",
        plan: sampling,
    },
    Workload {
        name: "constexpr_sampling",
        plan: constexpr_sampling,
    },
    Workload {
        name: "sampling_deep_152x152x6",
        plan: sampling_deep_152x152x6,
    },
    Workload {
        name: "sampling_deep_40x40x90",
        plan: sampling_deep_40x40x90,
    },
    Workload {
        name: "sampling_deep_64x64x16",
        plan: sampling_deep_64x64x16,
    },
    Workload {
        name: "sampling_deep_512x512x16",
        plan: sampling_deep_512x512x16,
    },
    Workload {
        name: "sampling_deep_4x16x252",
        plan: sampling_deep_4x16x252,
    },
    Workload {
        name: "sampling_deep_1x1x192",
        plan: sampling_deep_1x1x192,
    },
    Workload {
        name: "sampling_deep_512x512x16_8x8x1",
        plan: sampling_deep_512x512x16_8x8x1,
    },
    Workload {
        name: "sampling_deep_512x512x16_16x16x1",
        plan: sampling_deep_512x512x16_16x16x1,
    },
    Workload {
        name: "sampling_deep_512x512x16_4x4x4",
        plan: sampling_deep_512x512x16_4x4x4,
    },
    Workload {
        name: "sampling_distinct_64x64x16",
        plan: sampling_distinct_64x64x16,
    },
    Workload {
        name: "sampling_distinct_512x512x16",
        plan: sampling_distinct_512x512x16,
    },
    Workload {
        name: "wide_bindings",
        plan: wide_bindings,
    },
    Workload {
        name: "waves",
        plan: waves,
    },
    Workload {
        name: "vote_loop",
        plan: vote_loop,
    },
    Workload {
        name: "constant_managed",
        plan: constant_managed,
    },
    Workload {
        name: "constant_private",
        plan: constant_private,
    },
    Workload {
        name: "texture_read",
        plan: texture_read,
    },
    Workload {
        name: "draw_full",
        plan: draw_full,
    },
    Workload {
        name: "draw_2000",
        plan: draw_2000,
    },
    Workload {
        name: "bind_2000",
        plan: bind_2000,
    },
];

/// A creation workload: objects of one kind made and released in turn (`Session::create`).
#[derive(Clone, Copy, Debug)]
pub struct Create {
    /// Its name in reports and in `--only`.
    pub name: &'static str,
    /// What it makes.
    pub makes: Makes,
}

/// What a creation workload makes.
#[derive(Clone, Copy, Debug)]
pub enum Makes {
    /// `Shared` buffers of `length` bytes, each taking the range the last one left.
    Buffers { length: u64 },
    /// `Private` buffers of `length` bytes, each with its first use: one command buffer that
    /// copies its first and last 16 bytes into a `Shared` buffer, through to completion, the copy
    /// checked zero. The time holds the buffer's zeroing wherever the runtime does it: in the
    /// creation, or ahead of the first command buffer that uses it.
    PrivateBuffers { length: u64 },
    /// Compute pipeline states of one content: `function` of the authored library with the
    /// `float` function constants `constants` (index, value), values no other workload gives it,
    /// so the first creation is of a content the device has not seen and each later one of a
    /// content it has. [`STATE_CREATIONS`] are timed, whatever `--runs` says.
    States {
        function: &'static str,
        constants: &'static [(u32, f32)],
    },
    /// Render pipeline states of one content: the vertex function `vertex` and the fragment
    /// function `fragment` of the render library ([`render_library_air`]), drawing into one colour
    /// attachment of `color`. No other workload makes a render state, so the first creation is of
    /// a content the device has not seen and each later one of a content it has.
    /// [`STATE_CREATIONS`] are timed, whatever `--runs` says.
    RenderStates {
        vertex: &'static str,
        fragment: &'static str,
        color: PixelFormat,
    },
}

/// The kernel of a compute pipeline state row's warm-up
/// ([`CreateReport::device_first_us`](crate::run::CreateReport::device_first_us)): another of the
/// authored library's, bound as the row's own is (16 bytes it writes at buffer 0, a `uint` at
/// buffer 1). Some authored workloads make it too, so its state is the device's first only where
/// the creation rows run alone, as the matrix runs them.
pub const WARM_KERNEL: &str = "lcg_step";

/// The fragment function of a render pipeline state row's warm-up
/// ([`CreateReport::device_first_us`](crate::run::CreateReport::device_first_us)), with the row's
/// vertex function and attachment: another of the render library's.
pub const WARM_FRAGMENT: &str = "sampled_fragment";

/// The timed creations of a pipeline-state workload.
pub const STATE_CREATIONS: u32 = 1000;

/// The creation workloads.
pub const CREATES: [Create; 6] = [
    Create {
        name: "create_4KiB",
        makes: Makes::Buffers { length: 4 << 10 },
    },
    Create {
        name: "create_1MiB",
        makes: Makes::Buffers { length: 1 << 20 },
    },
    Create {
        name: "create_64MiB",
        makes: Makes::Buffers { length: 64 << 20 },
    },
    Create {
        name: "create_private_16MiB",
        makes: Makes::PrivateBuffers { length: 16 << 20 },
    },
    Create {
        name: "create_state",
        makes: Makes::States {
            function: "signed_zero_products",
            constants: &[(0, 2.0), (1, -2.0)],
        },
    },
    Create {
        name: "create_render_state",
        makes: Makes::RenderStates {
            vertex: "colored_vertex",
            fragment: "colored_fragment",
            color: PixelFormat::BGRA8Unorm,
        },
    },
];

/// The authored library's AIR bitcode: the fixture's LLVM text, assembled by `llvm-as`.
pub fn library_air() -> Result<Vec<u8>> {
    assemble("harness/workloads.ll")
}

/// The render library's AIR bitcode: the public fixture `examples/runtime_examples.ll`, whose
/// `colored_vertex` and `colored_fragment` draw a triangle list of flat colours, assembled by
/// `llvm-as`.
pub fn render_library_air() -> Result<Vec<u8>> {
    assemble("examples/runtime_examples.ll")
}

/// The public fixture at `file`, LLVM text, assembled by `llvm-as`.
fn assemble(file: &str) -> Result<Vec<u8>> {
    let ll = metal2vulkan_devtools::fixtures_dir().join(file);
    metal2vulkan_devtools::llvm_assemble(&ll)
        .map_err(|why| format!("llvm-as {}: {why:?}", ll.display()).into())
}

/// Fill-bound draws: 64 in one pass, each a triangle over the whole of a 512 by 512 target.
fn draw_full() -> Plan {
    draws("draw_full", 512, 64)
}

/// Draw-bound draws: 2,000 in one pass, each a triangle over the whole of a 16 by 16 target, so
/// each draw's own cost leads.
fn draw_2000() -> Plan {
    draws("draw_2000", 16, 2000)
}

/// The binds a draw makes, timed: 2,000 draws in one pass, as [`draw_2000`]'s, each with its
/// vertex function's positions (buffer 0) and a fragment texture (texture 0) set before it, both
/// alternating between two objects, so every bind is of an object the slot does not hold. The
/// fragment function reads no texture, so the draws' work is `draw_2000`'s; what the row adds is
/// the binds.
fn bind_2000() -> Plan {
    let mut plan = draws("bind_2000", 16, 2000);
    plan.buffers.push(plan.buffers[0].clone());
    plan.textures.push(TexturePlan::rgba16_float(4, 4, None));
    plan.textures.push(TexturePlan::rgba16_float(4, 4, None));
    for (at, dispatch) in plan.dispatches.iter_mut().enumerate() {
        let other = at % 2;
        dispatch
            .bindings
            .retain(|binding| !matches!(binding, Binding::Buffer { index: 0, .. }));
        dispatch.bindings.push(buffer(0, [0, 2][other]));
        dispatch.bindings.push(texture(0, 1 + other));
    }
    plan
}

/// `count` draws in one pass into a `side` by `side` `RGBA16Float` target, each of
/// `colored_vertex` and `colored_fragment` of the render library over one triangle that covers
/// the target, flat green: the target's every texel is green after the first run. The first
/// draw binds the vertex function's positions (buffer 0) and colours (buffer 1), which stay
/// bound for the rest, as Metal's bindings do.
fn draws(name: &str, side: u32, count: usize) -> Plan {
    const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
    let mut positions = vec![-1.0, -1.0, 3.0, -1.0, -1.0, 3.0];
    positions.resize(8, 0.0);
    let colors = [GREEN; 3].concat();
    let mut target = TexturePlan::rgba16_float(side, side, None);
    target.usage |= TextureUsage::RENDER_TARGET;
    let draw = |bindings| Dispatch {
        blits: Vec::new(),
        function: "colored_vertex".to_owned(),
        linked: Vec::new(),
        constants: Vec::new(),
        bindings,
        grid: Grid::Draw {
            fragment: "colored_fragment",
            target: 0,
            vertex_count: 3,
        },
    };
    let mut dispatches = vec![draw(vec![buffer(0, 0), buffer(1, 1)])];
    dispatches.resize_with(count, || draw(Vec::new()));
    let texels = side as usize * side as usize;
    Plan {
        name: name.to_owned(),
        library: Source::Render,
        buffers: vec![
            private(float_bytes(&positions)),
            private(float_bytes(&colors)),
        ],
        dispatches,
        expect: Vec::new(),
        textures: vec![target],
        expect_textures: vec![TextureExpect {
            texture: 0,
            values: Values::Floats {
                values: GREEN.repeat(texels),
                tolerance: 0.0,
            },
        }],
        outputs: Vec::new(),
        per_command_buffer: None,
        per_encoder: None,
        commit_to_complete: false,
        frames: false,
        scaled: None,
    }
}

/// Bandwidth-bound: 64 MiB of `float4`s read and 64 MiB written, `x * 2 + 1` each
/// (`scale_offset`), in `Private` storage. Every input is a whole number below 256.
fn bandwidth() -> Plan {
    const VECTORS: u64 = 4 << 20;
    let input: Vec<f32> = (words(1, 4 * VECTORS as usize).into_iter())
        .map(|word| (word & 255) as f32)
        .collect();
    let output = input.iter().map(|x| (x * 2.0 + 1.0).to_bits()).collect();
    Plan {
        name: "bandwidth".to_owned(),
        library: Source::Authored,
        buffers: vec![private(float_bytes(&input)), unset(16 * VECTORS)],
        dispatches: vec![Dispatch {
            blits: Vec::new(),
            function: "scale_offset".to_owned(),
            linked: Vec::new(),
            constants: Vec::new(),
            bindings: vec![
                buffer(0, 0),
                buffer(1, 1),
                bytes(2, float_bytes(&[2.0, 1.0])),
            ],
            grid: Grid::groups_of_256(VECTORS / 256),
        }],
        expect: vec![Expect {
            buffer: 1,
            values: Values::Words(output),
        }],
        textures: Vec::new(),
        expect_textures: Vec::new(),
        outputs: Vec::new(),
        per_command_buffer: None,
        per_encoder: None,
        commit_to_complete: false,
        frames: false,
        scaled: None,
    }
}

/// ALU-bound: 2^19 threads, each two dependent FMA chains of 1024 rounds (`fma_chains`). A
/// thread starts from its index's low ten bits, so the reference runs the 1024 starts.
fn alu() -> Plan {
    fma_chains("alu", 1024, 1)
}

/// Four long command buffers committed together, each one `fma_chains` dispatch of 2^19 threads,
/// and each command buffer's commit-to-complete time timed by its place in the run
/// (`Plan::commit_to_complete`): whether the first completes when it ends or waits for the ones
/// committed after it. GPUs differ about 150-fold in an `fma_chains` round, so no one length
/// keeps every GPU's command buffers long: the device sizes it ([`Scaled`]), from 1024 rounds a
/// dispatch doubled toward 10 ms of GPU time a command buffer.
fn long_4cb() -> Plan {
    Plan {
        scaled: Some(Scaled {
            build: long_4cb_at,
            base: 1024,
            target_us: 10_000.0,
        }),
        ..long_4cb_at(1024)
    }
}

/// `long_4cb` at `rounds` rounds a dispatch.
fn long_4cb_at(rounds: u32) -> Plan {
    Plan {
        commit_to_complete: true,
        frames: false,
        ..fma_chains("long_4cb", rounds, 4)
    }
}

/// `command_buffers` command buffers committed together, each one `fma_chains` dispatch of 2^19
/// threads, each thread two dependent FMA chains of `rounds` rounds. A thread starts from its
/// index's low ten bits, so the reference runs the 1024 starts.
fn fma_chains(name: &str, rounds: u32, command_buffers: usize) -> Plan {
    const THREADS: usize = 1 << 19;
    let ends: Vec<f32> = (0..1024u32)
        .map(|start| {
            let mut a = (start & 31) as f32 / 32.0;
            let mut b = ((start >> 5) & 31) as f32 / 32.0;
            for _ in 0..rounds {
                a = a.mul_add(0.9375, b * 0.0625);
                b = b.mul_add(0.875, a * 0.125);
            }
            a + b
        })
        .collect();
    let dispatch = Dispatch {
        blits: Vec::new(),
        function: "fma_chains".to_owned(),
        linked: Vec::new(),
        constants: Vec::new(),
        bindings: vec![buffer(0, 0), bytes(1, rounds.to_le_bytes().to_vec())],
        grid: Grid::groups_of_256(THREADS as u64 / 256),
    };
    Plan {
        name: name.to_owned(),
        library: Source::Authored,
        buffers: vec![unset(4 * THREADS as u64)],
        dispatches: vec![dispatch; command_buffers],
        expect: vec![Expect {
            buffer: 0,
            values: Values::Floats {
                values: (0..THREADS).map(|thread| ends[thread % 1024]).collect(),
                tolerance: 1e-4,
            },
        }],
        textures: Vec::new(),
        expect_textures: Vec::new(),
        outputs: Vec::new(),
        per_command_buffer: (command_buffers > 1).then_some(1),
        per_encoder: None,
        commit_to_complete: false,
        frames: false,
        scaled: None,
    }
}

/// Threadgroup memory with barriers: the inclusive prefix sums of each 256 of 8 Mi words
/// (`block_scan`, 16 barriers a threadgroup).
fn threadgroup_scan() -> Plan {
    const WORDS: usize = 8 << 20;
    let input = words(2, WORDS);
    let mut output = Vec::with_capacity(WORDS);
    for block in input.chunks(256) {
        let mut sum = 0u32;
        for &word in block {
            sum = sum.wrapping_add(word);
            output.push(sum);
        }
    }
    Plan {
        name: "threadgroup_scan".to_owned(),
        library: Source::Authored,
        buffers: vec![private(word_bytes(&input)), unset(4 * WORDS as u64)],
        dispatches: vec![Dispatch {
            blits: Vec::new(),
            function: "block_scan".to_owned(),
            linked: Vec::new(),
            constants: Vec::new(),
            bindings: vec![buffer(0, 0), buffer(1, 1)],
            grid: Grid::groups_of_256(WORDS as u64 / 256),
        }],
        expect: vec![Expect {
            buffer: 1,
            values: Values::Words(output),
        }],
        textures: Vec::new(),
        expect_textures: Vec::new(),
        outputs: Vec::new(),
        per_command_buffer: None,
        per_encoder: None,
        commit_to_complete: false,
        frames: false,
        scaled: None,
    }
}

/// SIMD reductions: the wrapping sum of 16 Mi words in two dispatches of `reduce_sum`, the first
/// a sum per 4096 words, the second the sum of those into a `Shared` buffer.
fn simd_reduce() -> Plan {
    const WORDS: u32 = 16 << 20;
    const CHUNK: u32 = 4096;
    const GROUPS: u32 = WORDS / CHUNK;
    let input = words(3, WORDS as usize);
    let sum = |words: &[u32]| words.iter().fold(0u32, |sum, &word| sum.wrapping_add(word));
    let partials: Vec<u32> = input.chunks(CHUNK as usize).map(sum).collect();
    let total = sum(&partials);
    let shape = |count: u32, chunk: u32| word_bytes(&[count, chunk]);
    Plan {
        name: "simd_reduce".to_owned(),
        library: Source::Authored,
        buffers: vec![
            private(word_bytes(&input)),
            unset(4 * GROUPS as u64),
            BufferPlan {
                length: 16,
                storage: Storage::Shared,
                contents: Some(vec![0; 16]),
                modified: None,
            },
        ],
        dispatches: vec![
            Dispatch {
                blits: Vec::new(),
                function: "reduce_sum".to_owned(),
                linked: Vec::new(),
                constants: Vec::new(),
                bindings: vec![buffer(0, 0), buffer(1, 1), bytes(2, shape(WORDS, CHUNK))],
                grid: Grid::groups_of_256(GROUPS as u64),
            },
            Dispatch {
                blits: Vec::new(),
                function: "reduce_sum".to_owned(),
                linked: Vec::new(),
                constants: Vec::new(),
                bindings: vec![buffer(0, 1), buffer(1, 2), bytes(2, shape(GROUPS, GROUPS))],
                grid: Grid::groups_of_256(1),
            },
        ],
        expect: vec![
            Expect {
                buffer: 1,
                values: Values::Words(partials),
            },
            Expect {
                buffer: 2,
                values: Values::Words(vec![total, 0, 0, 0]),
            },
        ],
        textures: Vec::new(),
        expect_textures: Vec::new(),
        outputs: Vec::new(),
        per_command_buffer: None,
        per_encoder: None,
        commit_to_complete: false,
        frames: false,
        scaled: None,
    }
}

/// 2D image-like indexing: a 3x3 binomial filter over a 1920x1080 image of `float4` texels
/// (`binomial3x3`), on a grid of one thread a texel in threadgroups of 16x16, the bottom row's
/// partial. Every input channel is a whole number below 256.
fn image_filter() -> Plan {
    const WIDTH: usize = 1920;
    const HEIGHT: usize = 1080;
    let input: Vec<f32> = (words(4, 4 * WIDTH * HEIGHT).into_iter())
        .map(|word| (word & 255) as f32)
        .collect();
    let at = |x: usize, y: usize, dx: isize, dy: isize| {
        let x = x.saturating_add_signed(dx).min(WIDTH - 1);
        let y = y.saturating_add_signed(dy).min(HEIGHT - 1);
        4 * (y * WIDTH + x)
    };
    let mut output = Vec::with_capacity(input.len());
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let mut sum = [0.0f32; 4];
            for dy in -1..=1isize {
                for dx in -1..=1isize {
                    let weight = (2 - dx.abs()) as f32 * (2 - dy.abs()) as f32;
                    let texel = at(x, y, dx, dy);
                    for (channel, sum) in sum.iter_mut().enumerate() {
                        *sum += input[texel + channel] * weight;
                    }
                }
            }
            output.extend(sum.map(|sum| (sum * 0.0625).to_bits()));
        }
    }
    let size = word_bytes(&[WIDTH as u32, HEIGHT as u32]);
    Plan {
        name: "image_filter".to_owned(),
        library: Source::Authored,
        buffers: vec![private(float_bytes(&input)), unset(4 * input.len() as u64)],
        dispatches: vec![Dispatch {
            blits: Vec::new(),
            function: "binomial3x3".to_owned(),
            linked: Vec::new(),
            constants: Vec::new(),
            bindings: vec![buffer(0, 0), buffer(1, 1), bytes(2, size)],
            grid: Grid::Threads {
                threads: [WIDTH as u64, HEIGHT as u64, 1],
                per_group: [16, 16, 1],
            },
        }],
        expect: vec![Expect {
            buffer: 1,
            values: Values::Words(output),
        }],
        textures: Vec::new(),
        expect_textures: Vec::new(),
        outputs: Vec::new(),
        per_command_buffer: None,
        per_encoder: None,
        commit_to_complete: false,
        frames: false,
        scaled: None,
    }
}

/// `row_cells` over a 2048 by 1024 grid, its width and height, whose rows are 2048 cells apart, in
/// threadgroups of 32 by 2, 64 by 8 and 512 by 1, four dispatches of each: each SIMD group lies
/// within one row of its threadgroup, so a runtime that runs a kernel's one-row variant there may
/// run these grids in its tiles.
fn row_cells() -> Plan {
    const WIDTH: usize = 2048;
    const HEIGHT: usize = 1024;
    let input: Vec<f32> = (words(7, WIDTH * HEIGHT).into_iter())
        .map(|word| (word & 1023) as f32)
        .collect();
    let mut output = Vec::with_capacity(3 * input.len());
    for &value in &input {
        output.extend([value, value * 2.0, value + 1.0].map(f32::to_bits));
    }
    let width = word_bytes(&[WIDTH as u32]);
    let height = word_bytes(&[HEIGHT as u32]);
    let dispatch = |per_group: [u64; 3]| Dispatch {
        blits: Vec::new(),
        function: "row_cells".to_owned(),
        linked: Vec::new(),
        constants: Vec::new(),
        bindings: vec![
            buffer(0, 0),
            buffer(1, 1),
            bytes(2, width.clone()),
            bytes(3, height.clone()),
            bytes(4, width.clone()),
        ],
        grid: Grid::Threads {
            threads: [WIDTH as u64, HEIGHT as u64, 1],
            per_group,
        },
    };
    let mut dispatches = Vec::new();
    for per_group in [[32, 2, 1], [64, 8, 1], [512, 1, 1]] {
        for _ in 0..4 {
            dispatches.push(dispatch(per_group));
        }
    }
    Plan {
        name: "row_cells".to_owned(),
        library: Source::Authored,
        buffers: vec![private(float_bytes(&input)), unset(4 * output.len() as u64)],
        dispatches,
        expect: vec![Expect {
            buffer: 1,
            values: Values::Words(output),
        }],
        textures: Vec::new(),
        expect_textures: Vec::new(),
        outputs: Vec::new(),
        per_command_buffer: None,
        per_encoder: None,
        commit_to_complete: false,
        frames: false,
        scaled: None,
    }
}

/// Many tiny dispatches in a chain: 512 dispatches of one threadgroup of `lcg_step` over the
/// same 256 words, each its own increment by `setBytes`, each reading what the one before wrote.
fn dispatch_chain() -> Plan {
    const WORDS: usize = 256;
    const STEPS: u32 = 512;
    let increment = |step: u32| 1_013_904_223u32.wrapping_add(step);
    let input = words(5, WORDS);
    let output = (input.iter())
        .map(|&word| {
            (0..STEPS).fold(word, |word, step| {
                word.wrapping_mul(1_664_525).wrapping_add(increment(step))
            })
        })
        .collect();
    let dispatches = (0..STEPS)
        .map(|step| {
            let set_bytes = bytes(1, increment(step).to_le_bytes().to_vec());
            Dispatch {
                blits: Vec::new(),
                function: "lcg_step".to_owned(),
                linked: Vec::new(),
                constants: Vec::new(),
                bindings: match step {
                    0 => vec![buffer(0, 0), set_bytes],
                    _ => vec![set_bytes],
                },
                grid: Grid::groups_of_256(1),
            }
        })
        .collect();
    Plan {
        name: "dispatch_chain".to_owned(),
        library: Source::Authored,
        buffers: vec![private(word_bytes(&input))],
        dispatches,
        expect: vec![Expect {
            buffer: 0,
            values: Values::Words(output),
        }],
        textures: Vec::new(),
        expect_textures: Vec::new(),
        outputs: Vec::new(),
        per_command_buffer: None,
        per_encoder: None,
        commit_to_complete: false,
        frames: false,
        scaled: None,
    }
}

fn word_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn float_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// A `Private` buffer starting with `contents`.
fn private(contents: Vec<u8>) -> BufferPlan {
    BufferPlan {
        length: contents.len() as u64,
        storage: Storage::Private,
        contents: Some(contents),
        modified: None,
    }
}

/// A `Private` buffer of `length` bytes whose first run writes before it reads.
fn unset(length: u64) -> BufferPlan {
    BufferPlan {
        length,
        storage: Storage::Private,
        contents: None,
        modified: None,
    }
}

/// A desktop session's frame, in the shape its recordings commit (a median of one command buffer
/// a frame, two at the 90th percentile, a few dispatches each), in compute alone: a run is one
/// command buffer of four `lcg_step` dispatches ([`lcg_frame`]). Its commit-to-complete time is
/// timed (`Plan::commit_to_complete`).
fn frame() -> Plan {
    Plan {
        commit_to_complete: true,
        frames: true,
        ..lcg_frame("frame", 1, 4, None, Storage::Managed)
    }
}

/// A game's median frame, in the shape its recordings commit (a median of five command buffers a
/// frame), in compute alone: a run is five command buffers of four `lcg_step` dispatches each,
/// `frame`'s command buffer five times over ([`lcg_frame`]), committed back to back and waited
/// for once. Each command buffer's commit-to-complete time is timed (`Plan::commit_to_complete`).
fn frame_5cb() -> Plan {
    Plan {
        commit_to_complete: true,
        frames: true,
        ..lcg_frame("frame_5cb", 5, 4, None, Storage::Managed)
    }
}

/// A burst of small command buffers: a run is 160 command buffers of two `lcg_step` dispatches
/// each ([`lcg_frame`]).
fn burst_160() -> Plan {
    lcg_frame("burst_160", 160, 2, None, Storage::Managed)
}

/// `burst_160` with its increment in a `Shared` buffer, kept for the record: the recordings bind
/// few small constant arguments from `Shared` memory ([`lcg_frame`]).
fn burst_160_shared() -> Plan {
    lcg_frame("burst_160_shared", 160, 2, None, Storage::Shared)
}

/// The burst a recording commits most: a run is 2,000 command buffers of one compute encoder of
/// one `lcg_step` dispatch each ([`lcg_frame`]).
fn burst_2000() -> Plan {
    lcg_frame("burst_2000", 2000, 1, None, Storage::Managed)
}

/// That burst at five compute encoders a command buffer: a run is 2,000 command buffers of five
/// compute encoders of one `lcg_step` dispatch each ([`lcg_frame`]).
fn burst_2000x5() -> Plan {
    lcg_frame("burst_2000x5", 2000, 5, Some(1), Storage::Managed)
}

/// A run of `command_buffers` command buffers of `per` tiny dispatches each, in compute encoders
/// of `per_encoder` dispatches (or one of them all), committed with no wait between and waited
/// for once: `lcg_step` over 256 words of `Private` storage, each dispatch setting its bindings,
/// since an encoder starts with none. Both are buffers (`setBuffer:offset:atIndex:`): the words,
/// and the increment in a buffer of `storage` written once, a `Managed` one then flushed with
/// `didModifyRange:`, as the recordings bind their small constant arguments; the run makes no
/// `setBytes` call (`dispatch_chain` measures those).
fn lcg_frame(
    name: &str,
    command_buffers: usize,
    per: usize,
    per_encoder: Option<usize>,
    storage: Storage,
) -> Plan {
    const WORDS: usize = 256;
    let steps = per * command_buffers;
    let increment = words(14, 1)[0];
    let input = words(13, WORDS);
    let output = (input.iter())
        .map(|&word| {
            (0..steps).fold(word, |word, _| {
                word.wrapping_mul(1_664_525).wrapping_add(increment)
            })
        })
        .collect();
    let dispatch = Dispatch {
        blits: Vec::new(),
        function: "lcg_step".to_owned(),
        linked: Vec::new(),
        constants: Vec::new(),
        bindings: vec![buffer(0, 0), buffer(1, 1)],
        grid: Grid::groups_of_256(1),
    };
    Plan {
        name: name.to_owned(),
        library: Source::Authored,
        buffers: vec![
            private(word_bytes(&input)),
            BufferPlan {
                length: 16,
                storage,
                contents: Some(word_bytes(&[increment, 0, 0, 0])),
                modified: None,
            },
        ],
        dispatches: vec![dispatch; steps],
        expect: vec![Expect {
            buffer: 0,
            values: Values::Words(output),
        }],
        textures: Vec::new(),
        expect_textures: Vec::new(),
        outputs: Vec::new(),
        per_command_buffer: Some(per),
        per_encoder,
        commit_to_complete: false,
        frames: false,
        scaled: None,
    }
}

/// Many buffers bound for each of many small dispatches: 256 dispatches of 256 threads of
/// `wide_bindings`, each binding 24 buffers, 23 inputs and the output, and each thread folding one
/// word of each input (`s · 3 + v`). Each dispatch reads 24 buffer addresses, which a runtime
/// passes in push constants as far as its push window holds them, and in a parameter block in
/// memory past it.
fn wide_bindings() -> Plan {
    const WORDS: usize = 256;
    const INPUTS: usize = 23;
    const DISPATCHES: usize = 256;
    let inputs: Vec<Vec<u32>> = (0..INPUTS as u64).map(|k| words(40 + k, WORDS)).collect();
    let output = (0..WORDS)
        .map(|i| {
            (1..INPUTS).fold(inputs[0][i], |s, k| {
                s.wrapping_mul(3).wrapping_add(inputs[k][i])
            })
        })
        .collect();
    let mut buffers: Vec<BufferPlan> = inputs.iter().map(|w| private(word_bytes(w))).collect();
    buffers.push(unset(4 * WORDS as u64));
    let dispatch = Dispatch {
        blits: Vec::new(),
        function: "wide_bindings".to_owned(),
        linked: Vec::new(),
        constants: Vec::new(),
        bindings: (0..=INPUTS).map(|k| buffer(k as u32, k)).collect(),
        grid: Grid::groups_of_256(1),
    };
    Plan {
        name: "wide_bindings".to_owned(),
        library: Source::Authored,
        buffers,
        dispatches: vec![dispatch; DISPATCHES],
        expect: vec![Expect {
            buffer: INPUTS,
            values: Values::Words(output),
        }],
        textures: Vec::new(),
        expect_textures: Vec::new(),
        outputs: Vec::new(),
        per_command_buffer: None,
        per_encoder: None,
        commit_to_complete: false,
        frames: false,
        scaled: None,
    }
}

/// The fixed cost of a wave: one dispatch of 175 by 4,200 threadgroups of 32 threads
/// (`wave_store`), 735,000 one-wave threadgroups, each thread one load from a constant buffer
/// argument (a 16-byte `Private` buffer) and one 4-byte store of `base + index` to a `Private`
/// buffer of 94 MB. A run's GPU time over its 735,000 waves is a per-wave budget: what each wave
/// costs to start and end, beside its 128 bytes of stores.
fn waves() -> Plan {
    const GROUPS: [u64; 2] = [175, 4200];
    const WIDTH: u64 = 32 * GROUPS[0];
    const THREADS: u64 = WIDTH * GROUPS[1];
    const BASE: u32 = 0x9e37_79b9;
    let output = (0..THREADS as u32)
        .map(|index| BASE.wrapping_add(index))
        .collect();
    Plan {
        name: "waves".to_owned(),
        library: Source::Authored,
        buffers: vec![unset(4 * THREADS), private(word_bytes(&[BASE, 0, 0, 0]))],
        dispatches: vec![Dispatch {
            blits: Vec::new(),
            function: "wave_store".to_owned(),
            linked: Vec::new(),
            constants: Vec::new(),
            bindings: vec![buffer(0, 0), buffer(1, 1)],
            grid: Grid::Threadgroups {
                groups: [GROUPS[0], GROUPS[1], 1],
                threads: [32, 1, 1],
            },
        }],
        expect: vec![Expect {
            buffer: 0,
            values: Values::Words(output),
        }],
        textures: Vec::new(),
        expect_textures: Vec::new(),
        outputs: Vec::new(),
        per_command_buffer: None,
        per_encoder: None,
        commit_to_complete: false,
        frames: false,
        scaled: None,
    }
}

/// A SIMD-group barrier inside a loop at one SIMD group per threadgroup: 2^20 threads in
/// threadgroups of 32 (`vote_loop`), every thread of a threadgroup stepping an LCG its
/// threadgroup's count of rounds (below 256), with a `simdgroup_barrier` in each round, and
/// writing its state and its lane. The count is read per thread, so the translation cannot prove
/// it uniform: on a GPU whose subgroups are 64 wide, a threadgroup is one wave running the
/// kernel's emulated SIMD group, which votes on the loop's exit once a round.
fn vote_loop() -> Plan {
    const THREADS: u32 = 1 << 20;
    const GROUP: u32 = 32;
    let counts = words(15, (THREADS / GROUP) as usize);
    let steps: Vec<u32> = (0..THREADS)
        .map(|index| counts[(index / GROUP) as usize] & 255)
        .collect();
    let ends: Vec<u32> = (0..THREADS)
        .flat_map(|index| {
            let state = (0..steps[index as usize]).fold(index, |state, _| {
                state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223)
            });
            [state, index % GROUP]
        })
        .collect();
    Plan {
        name: "vote_loop".to_owned(),
        library: Source::Authored,
        buffers: vec![private(word_bytes(&steps)), unset(8 * u64::from(THREADS))],
        dispatches: vec![Dispatch {
            blits: Vec::new(),
            function: "vote_loop".to_owned(),
            linked: Vec::new(),
            constants: Vec::new(),
            bindings: vec![buffer(0, 0), buffer(1, 1)],
            grid: Grid::Threadgroups {
                groups: [u64::from(THREADS / GROUP), 1, 1],
                threads: [u64::from(GROUP), 1, 1],
            },
        }],
        expect: vec![Expect {
            buffer: 1,
            values: Values::Words(ends),
        }],
        textures: Vec::new(),
        expect_textures: Vec::new(),
        outputs: Vec::new(),
        per_command_buffer: None,
        per_encoder: None,
        commit_to_complete: false,
        frames: false,
        scaled: None,
    }
}

/// Reads of a 64 KiB table as constants (`constant_reads`): 2^20 threads in threadgroups of 256,
/// each folding 64 of the table's 4,096 entries into a word, the table `Managed`, and the first
/// 256 bytes of it written again by the CPU before each run and marked with `didModifyRange:`, as
/// a program updates a constant buffer a little before each frame. On a GPU with memory of its
/// own, each run reads the table after its update reached the GPU's copy.
fn constant_managed() -> Plan {
    constant_reads("constant_managed", Storage::Managed, Some(0..256))
}

/// [`constant_managed`]'s reads from a `Private` table that nothing writes after its first run:
/// the reference its time is read against.
fn constant_private() -> Plan {
    constant_reads("constant_private", Storage::Private, None)
}

/// The constant reads of [`constant_managed`] from a table of `storage`, `modified` before each
/// run.
fn constant_reads(name: &str, storage: Storage, modified: Option<Range<u64>>) -> Plan {
    const THREADS: u32 = 1 << 20;
    const ENTRIES: u32 = 4096;
    const GROUP: u32 = 256;
    let table = words(16, 4 * ENTRIES as usize);
    let folded: Vec<u32> = (0..THREADS)
        .map(|index| {
            (0..64).fold(index, |folded, i| {
                let at = 4 * ((index + i * 67) & (ENTRIES - 1)) as usize;
                let entry = &table[at..at + 4];
                (folded ^ entry[0] ^ entry[1]).wrapping_add(entry[2] ^ entry[3])
            })
        })
        .collect();
    Plan {
        name: name.to_owned(),
        library: Source::Authored,
        buffers: vec![
            BufferPlan {
                length: 16 * u64::from(ENTRIES),
                storage,
                contents: Some(word_bytes(&table)),
                modified,
            },
            unset(4 * u64::from(THREADS)),
        ],
        dispatches: vec![Dispatch {
            blits: Vec::new(),
            function: "constant_reads".to_owned(),
            linked: Vec::new(),
            constants: Vec::new(),
            bindings: vec![buffer(0, 0), buffer(1, 1)],
            grid: Grid::Threadgroups {
                groups: [u64::from(THREADS / GROUP), 1, 1],
                threads: [u64::from(GROUP), 1, 1],
            },
        }],
        expect: vec![Expect {
            buffer: 1,
            values: Values::Words(folded),
        }],
        textures: Vec::new(),
        expect_textures: Vec::new(),
        outputs: Vec::new(),
        per_command_buffer: None,
        per_encoder: None,
        commit_to_complete: false,
        frames: false,
        scaled: None,
    }
}

/// Narrow integers: 2^19 threads, each 512 steps of 8- and 16-bit arithmetic (`narrow_mix`), every
/// value held in a 32-bit container whose bits above its width the translation clears after each
/// step that can set them. The reference runs each thread's steps in Rust's wrapping arithmetic.
fn narrow_ints() -> Plan {
    const THREADS: u32 = 1 << 19;
    const ROUNDS: u32 = 512;
    let output = (0..THREADS)
        .map(|index| {
            let mut a = index.wrapping_mul(40503) as u16;
            let mut b = (index ^ 0x5a5a) as i16;
            let mut c = index as u8;
            let mut d = (index >> 3) as i8;
            for round in 0..ROUNDS {
                a = a.wrapping_mul(25173).wrapping_add(13849);
                b = b.wrapping_add((a >> 3) as i16);
                c = c.wrapping_mul(5).wrapping_add(b as u8);
                d = d.wrapping_add(c as i8).wrapping_sub(round as i8);
            }
            u32::from(a) ^ (u32::from(b as u16) << 16) ^ u32::from(c) ^ (u32::from(d as u8) << 8)
        })
        .collect();
    Plan {
        name: "narrow_ints".to_owned(),
        library: Source::Authored,
        buffers: vec![unset(4 * u64::from(THREADS))],
        dispatches: vec![Dispatch {
            blits: Vec::new(),
            function: "narrow_mix".to_owned(),
            linked: Vec::new(),
            constants: Vec::new(),
            bindings: vec![buffer(0, 0), bytes(1, ROUNDS.to_le_bytes().to_vec())],
            grid: Grid::groups_of_256(u64::from(THREADS) / 256),
        }],
        expect: vec![Expect {
            buffer: 0,
            values: Values::Words(output),
        }],
        textures: Vec::new(),
        expect_textures: Vec::new(),
        outputs: Vec::new(),
        per_command_buffer: None,
        per_encoder: None,
        commit_to_complete: false,
        frames: false,
        scaled: None,
    }
}

/// Products by a constant zero of either sign: 2^19 threads, each 512 steps that multiply a
/// `float4` by the function constants +0.0 and -0.0 in one block beside an FMA
/// (`signed_zero_products`, in precise arithmetic). The products' sum is a zero, so the result is
/// the FMA chain's, held to a tolerance.
fn zero_products() -> Plan {
    const THREADS: u32 = 1 << 19;
    const ROUNDS: u32 = 512;
    let ends: Vec<[f32; 4]> = (0..4096u32)
        .map(|index| {
            let mut a = [
                (index & 63) as f32,
                -(((index >> 6) & 63) as f32),
                0.5,
                -0.25,
            ];
            for _ in 0..ROUNDS {
                for x in &mut a {
                    *x = x.mul_add(0.96875, 0.03125);
                }
            }
            a
        })
        .collect();
    let float = |index, value: f32| Constant {
        index,
        data_type: 3,
        value: value.to_le_bytes().to_vec(),
    };
    Plan {
        name: "zero_products".to_owned(),
        library: Source::Authored,
        buffers: vec![unset(16 * u64::from(THREADS))],
        dispatches: vec![Dispatch {
            blits: Vec::new(),
            function: "signed_zero_products".to_owned(),
            linked: Vec::new(),
            constants: vec![float(0, 0.0), float(1, -0.0)],
            bindings: vec![buffer(0, 0), bytes(1, ROUNDS.to_le_bytes().to_vec())],
            grid: Grid::groups_of_256(u64::from(THREADS) / 256),
        }],
        expect: vec![Expect {
            buffer: 0,
            values: Values::Floats {
                values: (0..THREADS)
                    .flat_map(|thread| ends[(thread % 4096) as usize])
                    .collect(),
                tolerance: 1e-4,
            },
        }],
        textures: Vec::new(),
        expect_textures: Vec::new(),
        outputs: Vec::new(),
        per_command_buffer: None,
        per_encoder: None,
        commit_to_complete: false,
        frames: false,
        scaled: None,
    }
}

/// `setBuffer:offset:atIndex:` of the plan's buffer `buffer` at its start.
fn buffer(index: u32, buffer: usize) -> Binding {
    Binding::Buffer {
        index,
        buffer,
        offset: 0,
    }
}

/// `setBytes:length:atIndex:`.
fn bytes(index: u32, bytes: Vec<u8>) -> Binding {
    Binding::Bytes { index, bytes }
}

/// `setTexture:atIndex:` of the plan's texture `texture`.
fn texture(index: u32, texture: usize) -> Binding {
    Binding::Texture { index, texture }
}

/// How far a sampled texel may be from its reference. A filter's weights and the conversion to
/// `RGBA16Float` are the device's own; the sampling workloads' inputs make each result exact in
/// `half` under weights of halves and quarters, so this allows two last-place steps of a `half`
/// below 4.
const TEXEL_TOLERANCE: f32 = 1.0 / 256.0;

/// A `side` by `side` texture's starting texels ([`sixteenths_of`]).
fn sixteenths(seed: u64, side: u32) -> Vec<f32> {
    sixteenths_of(seed, side, side)
}

/// A `width` by `height` texture's starting texels, four channels each, every one a sixteenth of
/// a whole number below 64: exact in `half`.
fn sixteenths_of(seed: u64, width: u32, height: u32) -> Vec<f32> {
    let count = 4 * width as usize * height as usize;
    (words(seed, count).into_iter())
        .map(|word| (word & 63) as f32 / 16.0)
        .collect()
}

/// One thread a texel of a `side` by `side` texture, in threadgroups of 16x16.
fn texel_grid(side: u32) -> Grid {
    let tiles = u64::from(side / 16);
    Grid::Threadgroups {
        groups: [tiles, tiles, 1],
        threads: [16, 16, 1],
    }
}

/// A sampling workload's plan: `function` from a `from`-sided texture of `source` texels into a
/// `to`-sided one, which must then hold `expected`.
fn sampling_plan(
    name: &str,
    function: &str,
    (from, source): (u32, Vec<f32>),
    (to, expected): (u32, Vec<f32>),
    sampler: Option<Filter>,
) -> Plan {
    let mut bindings = vec![texture(0, 0), texture(1, 1)];
    bindings.extend(sampler.map(|filter| Binding::Sampler {
        index: 0,
        sampler: SamplerPlan::clamped(filter),
    }));
    Plan {
        name: name.to_owned(),
        library: Source::Authored,
        buffers: Vec::new(),
        textures: vec![
            TexturePlan::rgba16_float(from, from, Some(source)),
            TexturePlan::rgba16_float(to, to, None),
        ],
        dispatches: vec![Dispatch {
            blits: Vec::new(),
            function: function.to_owned(),
            linked: Vec::new(),
            constants: Vec::new(),
            bindings,
            grid: texel_grid(to),
        }],
        expect: Vec::new(),
        expect_textures: vec![TextureExpect {
            texture: 1,
            values: Values::Floats {
                values: expected,
                tolerance: TEXEL_TOLERANCE,
            },
        }],
        outputs: Vec::new(),
        per_command_buffer: None,
        per_encoder: None,
        commit_to_complete: false,
        frames: false,
        scaled: None,
    }
}

/// Sampling through a runtime sampler (`setSamplerState:atIndex:`, linear, clamped to the edge):
/// a 2x downsample of a 2048x2048 `RGBA16Float` texture into a 1024x1024 one (`sample_down`),
/// each texel one sample at the corner its four source texels share, one texture write a thread.
/// Each filter weight is one half, so each texel is the mean of its four source texels.
fn sampling() -> Plan {
    const SIDE: u32 = 2048;
    let source = sixteenths(15, SIDE);
    let half = SIDE / 2;
    let at = |x: u32, y: u32, channel: u32| source[(4 * (y * SIDE + x) + channel) as usize];
    let expected = (0..half)
        .flat_map(|y| (0..half).flat_map(move |x| (0..4).map(move |c| (x, y, c))))
        .map(|(x, y, c)| {
            let (x, y) = (2 * x, 2 * y);
            (at(x, y, c) + at(x + 1, y, c) + at(x, y + 1, c) + at(x + 1, y + 1, c)) / 4.0
        })
        .collect();
    sampling_plan(
        "sampling",
        "sample_down",
        (SIDE, source),
        (half, expected),
        Some(Filter::Linear),
    )
}

/// Sampling through a constexpr sampler (linear, clamped to the edge): a 2x upsample of a
/// 1024x1024 `RGBA16Float` texture into a 2048x2048 one (`sample_up`), each texel one sample a
/// quarter of a source texel from a source texel's center, one texture write a thread. Each
/// filter weight is a quarter or three quarters.
fn constexpr_sampling() -> Plan {
    const SIDE: u32 = 1024;
    let source = sixteenths(16, SIDE);
    // A destination coordinate's two source texels along its axis, and the second's weight: the
    // sample is at `x / 2 - 1/4` source texels from the first texel's center, clamped to the edge.
    let taps = |x: u32| {
        let k = x / 2;
        if x % 2 == 0 {
            (k.saturating_sub(1), k, 0.75f32)
        } else {
            (k, (k + 1).min(SIDE - 1), 0.25f32)
        }
    };
    let at = |x: u32, y: u32, channel: u32| source[(4 * (y * SIDE + x) + channel) as usize];
    let double = 2 * SIDE;
    let expected = (0..double)
        .flat_map(|y| (0..double).flat_map(move |x| (0..4).map(move |c| (x, y, c))))
        .map(|(x, y, c)| {
            let ((x0, x1, wx), (y0, y1, wy)) = (taps(x), taps(y));
            let row = |y| at(x0, y, c) * (1.0 - wx) + at(x1, y, c) * wx;
            row(y0) * (1.0 - wy) + row(y1) * wy
        })
        .collect();
    sampling_plan(
        "constexpr_sampling",
        "sample_up",
        (SIDE, source),
        (double, expected),
        None,
    )
}

/// Sampling over a grid more than one deep (`sample_down_layers`, through a runtime sampler,
/// linear, clamped to the edge): a grid `x` by `y` by `z` in threadgroups of 8x8x4, each layer a
/// 2x downsample of a `2t` by `2t` `RGBA16Float` texture into its own `t` by `t` rows of the
/// destination (`t` the larger of `x` and `y` rounded up to whole 16x16 tiles, as the harness's
/// texel kernels need), one texture write a thread: each texel in the grid the mean of its four
/// source texels, the rest left zero. The shapes are those a recorded stream's image-only kernels
/// ran deep grids at, thin and ragged ones among them.
fn sampling_deep(name: &str, threads: [u32; 3]) -> Plan {
    deep_sampling(name, threads, [8, 8, 4], false)
}

/// [`sampling_deep`] in threadgroups of `per_group`; with `distinct`, through
/// `sample_down_distinct`, whose layer `z` downsamples its own `2t` by `2t` rows of a `2t` by
/// `2t·z` source, so no two layers read the same texels.
fn deep_sampling(name: &str, [x, y, z]: [u32; 3], per_group: [u32; 3], distinct: bool) -> Plan {
    let t = x.max(y).next_multiple_of(16);
    let side = 2 * t;
    let layers = if distinct { z } else { 1 };
    let source = sixteenths_of(18, side, side * layers);
    let at = |i: u32, j: u32, c: u32| source[(4 * (j * side + i) + c) as usize];
    let layer = |k: u32| -> Vec<f32> {
        let rows = 2 * t * k.min(layers - 1);
        (0..t)
            .flat_map(|j| (0..t).flat_map(move |i| (0..4).map(move |c| (i, j, c))))
            .map(|(i, j, c)| match i < x && j < y {
                true => {
                    let (i, j) = (2 * i, rows + 2 * j);
                    (at(i, j, c) + at(i + 1, j, c) + at(i, j + 1, c) + at(i + 1, j + 1, c)) / 4.0
                }
                false => 0.0,
            })
            .collect()
    };
    let expected = match distinct {
        true => (0..z).flat_map(layer).collect(),
        false => {
            let one = layer(0);
            (0..z).flat_map(|_| one.iter().copied()).collect()
        }
    };
    let function = match distinct {
        true => "sample_down_distinct",
        false => "sample_down_layers",
    };
    let mut plan = sampling_plan(
        name,
        function,
        (16, Vec::new()),
        (16, expected),
        Some(Filter::Linear),
    );
    plan.textures = vec![
        TexturePlan::rgba16_float(side, side * layers, Some(source)),
        TexturePlan::rgba16_float(t, t * z, Some(vec![0.0; 4 * (t * t * z) as usize])),
    ];
    plan.dispatches[0].grid = Grid::Threads {
        threads: [u64::from(x), u64::from(y), u64::from(z)],
        per_group: per_group.map(u64::from),
    };
    plan
}

/// [`sampling_deep`] over a 152x152x6 grid.
fn sampling_deep_152x152x6() -> Plan {
    sampling_deep("sampling_deep_152x152x6", [152, 152, 6])
}

/// [`sampling_deep`] over a 40x40x90 grid.
fn sampling_deep_40x40x90() -> Plan {
    sampling_deep("sampling_deep_40x40x90", [40, 40, 90])
}

/// [`sampling_deep`] over a 64x64x16 grid.
fn sampling_deep_64x64x16() -> Plan {
    sampling_deep("sampling_deep_64x64x16", [64, 64, 16])
}

/// [`sampling_deep`] over a 512x512x16 grid.
fn sampling_deep_512x512x16() -> Plan {
    sampling_deep("sampling_deep_512x512x16", [512, 512, 16])
}

/// [`sampling_deep`] over a 4x16x252 grid.
fn sampling_deep_4x16x252() -> Plan {
    sampling_deep("sampling_deep_4x16x252", [4, 16, 252])
}

/// [`sampling_deep`] over a 1x1x192 grid.
fn sampling_deep_1x1x192() -> Plan {
    sampling_deep("sampling_deep_1x1x192", [1, 1, 192])
}

/// [`sampling_deep`] over a 512x512x16 grid in threadgroups of 8x8x1.
fn sampling_deep_512x512x16_8x8x1() -> Plan {
    deep_sampling(
        "sampling_deep_512x512x16_8x8x1",
        [512, 512, 16],
        [8, 8, 1],
        false,
    )
}

/// [`sampling_deep`] over a 512x512x16 grid in threadgroups of 16x16x1.
fn sampling_deep_512x512x16_16x16x1() -> Plan {
    deep_sampling(
        "sampling_deep_512x512x16_16x16x1",
        [512, 512, 16],
        [16, 16, 1],
        false,
    )
}

/// [`sampling_deep`] over a 512x512x16 grid in threadgroups of 4x4x4.
fn sampling_deep_512x512x16_4x4x4() -> Plan {
    deep_sampling(
        "sampling_deep_512x512x16_4x4x4",
        [512, 512, 16],
        [4, 4, 4],
        false,
    )
}

/// [`deep_sampling`] of distinct layers over a 64x64x16 grid in threadgroups of 8x8x4.
fn sampling_distinct_64x64x16() -> Plan {
    deep_sampling("sampling_distinct_64x64x16", [64, 64, 16], [8, 8, 4], true)
}

/// [`deep_sampling`] of distinct layers over a 512x512x16 grid in threadgroups of 8x8x4.
fn sampling_distinct_512x512x16() -> Plan {
    deep_sampling(
        "sampling_distinct_512x512x16",
        [512, 512, 16],
        [8, 8, 4],
        true,
    )
}

/// A texture read out to a buffer: each thread of a 2048x2048 grid, in threadgroups of 16x16,
/// reads its texel of an `RGBA16Float` texture and stores it as a `float4` at `y * 2048 + x`
/// (`texels_out`), 64 MiB stored. The kernel reads its position in the grid and nothing of its
/// threadgroup, so a runtime may run it in workgroups of any shape; on some drivers a store per
/// thread through a device address takes twice as long in 16x16 workgroups as in wide ones.
fn texture_read() -> Plan {
    const SIDE: u32 = 2048;
    let texels = sixteenths(17, SIDE);
    let stored = texels.iter().map(|texel| texel.to_bits()).collect();
    Plan {
        name: "texture_read".to_owned(),
        library: Source::Authored,
        buffers: vec![unset(16 * u64::from(SIDE) * u64::from(SIDE))],
        textures: vec![TexturePlan::rgba16_float(SIDE, SIDE, Some(texels))],
        dispatches: vec![Dispatch {
            blits: Vec::new(),
            function: "texels_out".to_owned(),
            linked: Vec::new(),
            constants: Vec::new(),
            bindings: vec![texture(0, 0), buffer(0, 0), bytes(1, word_bytes(&[SIDE]))],
            grid: texel_grid(SIDE),
        }],
        expect: vec![Expect {
            buffer: 0,
            values: Values::Words(stored),
        }],
        expect_textures: Vec::new(),
        outputs: Vec::new(),
        per_command_buffer: None,
        per_encoder: None,
        commit_to_complete: false,
        frames: false,
        scaled: None,
    }
}
