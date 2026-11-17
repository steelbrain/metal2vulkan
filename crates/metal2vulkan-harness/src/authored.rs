//! The authored workloads: the kernels of the public fixture `harness/workloads.metal`, their
//! inputs made here from a fixed seed, and the CPU reference of their results. Every result is
//! exact on any device but the ALU workload's, whose `float` chains are held to a tolerance.

use crate::plan::{
    seeded_words as words, Binding, BufferPlan, Constant, Dispatch, Expect, Filter, Grid, Plan,
    Source, TextureExpect, TexturePlan, Values,
};
use crate::{Result, Storage};

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
pub const WORKLOADS: [Workload; 15] = [
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
        name: "frame",
        plan: frame,
    },
    Workload {
        name: "burst_160",
        plan: burst_160,
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
        name: "wide_bindings",
        plan: wide_bindings,
    },
];

/// A creation workload: `Shared` buffers of one length made and released in turn, so that each
/// takes the range the last one left (`Session::create`).
#[derive(Clone, Copy, Debug)]
pub struct Create {
    /// Its name in reports and in `--only`.
    pub name: &'static str,
    /// Each buffer's length in bytes.
    pub length: u64,
}

/// The creation workloads.
pub const CREATES: [Create; 3] = [
    Create {
        name: "create_4KiB",
        length: 4 << 10,
    },
    Create {
        name: "create_1MiB",
        length: 1 << 20,
    },
    Create {
        name: "create_64MiB",
        length: 64 << 20,
    },
];

/// The authored library's AIR bitcode: the fixture's LLVM text, assembled by `llvm-as`.
pub fn library_air() -> Result<Vec<u8>> {
    let ll = metal2vulkan_devtools::fixtures_dir().join("harness/workloads.ll");
    metal2vulkan_devtools::llvm_assemble(&ll)
        .map_err(|why| format!("llvm-as {}: {why:?}", ll.display()).into())
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
            function: "scale_offset".to_owned(),
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
    }
}

/// ALU-bound: 2^19 threads, each two dependent FMA chains of 1024 rounds (`fma_chains`). A
/// thread starts from its index's low ten bits, so the reference runs the 1024 starts.
fn alu() -> Plan {
    const THREADS: usize = 1 << 19;
    const ROUNDS: u32 = 1024;
    let ends: Vec<f32> = (0..1024u32)
        .map(|start| {
            let mut a = (start & 31) as f32 / 32.0;
            let mut b = ((start >> 5) & 31) as f32 / 32.0;
            for _ in 0..ROUNDS {
                a = a.mul_add(0.9375, b * 0.0625);
                b = b.mul_add(0.875, a * 0.125);
            }
            a + b
        })
        .collect();
    Plan {
        name: "alu".to_owned(),
        library: Source::Authored,
        buffers: vec![unset(4 * THREADS as u64)],
        dispatches: vec![Dispatch {
            function: "fma_chains".to_owned(),
            constants: Vec::new(),
            bindings: vec![buffer(0, 0), bytes(1, ROUNDS.to_le_bytes().to_vec())],
            grid: Grid::groups_of_256(THREADS as u64 / 256),
        }],
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
        per_command_buffer: None,
        per_encoder: None,
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
            function: "block_scan".to_owned(),
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
            },
        ],
        dispatches: vec![
            Dispatch {
                function: "reduce_sum".to_owned(),
                constants: Vec::new(),
                bindings: vec![buffer(0, 0), buffer(1, 1), bytes(2, shape(WORDS, CHUNK))],
                grid: Grid::groups_of_256(GROUPS as u64),
            },
            Dispatch {
                function: "reduce_sum".to_owned(),
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
            function: "binomial3x3".to_owned(),
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
                function: "lcg_step".to_owned(),
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
    }
}

/// A `Shared` buffer starting with `contents`, written through `contents`.
fn shared(contents: Vec<u8>) -> BufferPlan {
    BufferPlan {
        length: contents.len() as u64,
        storage: Storage::Shared,
        contents: Some(contents),
    }
}

/// A `Private` buffer of `length` bytes whose first run writes before it reads.
fn unset(length: u64) -> BufferPlan {
    BufferPlan {
        length,
        storage: Storage::Private,
        contents: None,
    }
}

/// A desktop session's frame, in the shape its recordings commit (a median of one command buffer
/// a frame, two at the 90th percentile, a few dispatches each), in compute alone: a run is one
/// command buffer of four `lcg_step` dispatches ([`lcg_frame`]).
fn frame() -> Plan {
    lcg_frame("frame", 1, 4, None)
}

/// A burst of small command buffers: a run is 160 command buffers of two `lcg_step` dispatches
/// each ([`lcg_frame`]).
fn burst_160() -> Plan {
    lcg_frame("burst_160", 160, 2, None)
}

/// The burst a recording commits most: a run is 2,000 command buffers of one compute encoder of
/// one `lcg_step` dispatch each ([`lcg_frame`]).
fn burst_2000() -> Plan {
    lcg_frame("burst_2000", 2000, 1, None)
}

/// That burst at five compute encoders a command buffer: a run is 2,000 command buffers of five
/// compute encoders of one `lcg_step` dispatch each ([`lcg_frame`]).
fn burst_2000x5() -> Plan {
    lcg_frame("burst_2000x5", 2000, 5, Some(1))
}

/// A run of `command_buffers` command buffers of `per` tiny dispatches each, in compute encoders
/// of `per_encoder` dispatches (or one of them all), committed with no wait between and waited
/// for once: `lcg_step` over 256 words of `Private` storage, each dispatch setting its bindings,
/// since an encoder starts with none. Both are buffers (`setBuffer:offset:atIndex:`): the words,
/// and the increment in a `Shared` buffer written once; the run makes no `setBytes` call
/// (`dispatch_chain` measures those).
fn lcg_frame(name: &str, command_buffers: usize, per: usize, per_encoder: Option<usize>) -> Plan {
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
        function: "lcg_step".to_owned(),
        constants: Vec::new(),
        bindings: vec![buffer(0, 0), buffer(1, 1)],
        grid: Grid::groups_of_256(1),
    };
    Plan {
        name: name.to_owned(),
        library: Source::Authored,
        buffers: vec![
            private(word_bytes(&input)),
            shared(word_bytes(&[increment, 0, 0, 0])),
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
        function: "wide_bindings".to_owned(),
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
            function: "narrow_mix".to_owned(),
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
            function: "signed_zero_products".to_owned(),
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

/// A `side` by `side` texture's starting texels, four channels each, every one a sixteenth of a
/// whole number below 64: exact in `half`.
fn sixteenths(seed: u64, side: u32) -> Vec<f32> {
    let count = 4 * side as usize * side as usize;
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
    bindings.extend(sampler.map(|filter| Binding::Sampler { index: 0, filter }));
    Plan {
        name: name.to_owned(),
        library: Source::Authored,
        buffers: Vec::new(),
        textures: vec![
            TexturePlan {
                width: from,
                height: from,
                texels: Some(source),
            },
            TexturePlan {
                width: to,
                height: to,
                texels: None,
            },
        ],
        dispatches: vec![Dispatch {
            function: function.to_owned(),
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
