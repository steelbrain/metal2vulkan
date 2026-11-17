//! The authored workloads: the kernels of the public fixture `harness/workloads.metal`, their
//! inputs made here from a fixed seed, and the CPU reference of their results. Every result is
//! exact on any device but the ALU workload's, whose `float` chains are held to a tolerance.

use crate::plan::{Binding, BufferPlan, Dispatch, Expect, Grid, Plan, Values};
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
pub const WORKLOADS: [Workload; 6] = [
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
        buffers: vec![private(float_bytes(&input)), unset(16 * VECTORS)],
        dispatches: vec![Dispatch {
            function: "scale_offset".to_owned(),
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
        buffers: vec![unset(4 * THREADS as u64)],
        dispatches: vec![Dispatch {
            function: "fma_chains".to_owned(),
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
        buffers: vec![private(word_bytes(&input)), unset(4 * WORDS as u64)],
        dispatches: vec![Dispatch {
            function: "block_scan".to_owned(),
            bindings: vec![buffer(0, 0), buffer(1, 1)],
            grid: Grid::groups_of_256(WORDS as u64 / 256),
        }],
        expect: vec![Expect {
            buffer: 1,
            values: Values::Words(output),
        }],
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
                bindings: vec![buffer(0, 0), buffer(1, 1), bytes(2, shape(WORDS, CHUNK))],
                grid: Grid::groups_of_256(GROUPS as u64),
            },
            Dispatch {
                function: "reduce_sum".to_owned(),
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
        buffers: vec![private(float_bytes(&input)), unset(4 * input.len() as u64)],
        dispatches: vec![Dispatch {
            function: "binomial3x3".to_owned(),
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
        buffers: vec![private(word_bytes(&input))],
        dispatches,
        expect: vec![Expect {
            buffer: 0,
            values: Values::Words(output),
        }],
    }
}

/// `count` words of SplitMix64 from `seed`, the low half of each output.
fn words(seed: u64, count: usize) -> Vec<u32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            (z ^ (z >> 31)) as u32
        })
        .collect()
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

/// A `Private` buffer of `length` bytes whose first run writes before it reads.
fn unset(length: u64) -> BufferPlan {
    BufferPlan {
        length,
        storage: Storage::Private,
        contents: None,
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
