//! The index rewrite pass: its modules valid on every committed profile (L3, `G-val`), and, on
//! the CPU interpreter, the list it writes is the one Metal draws: a strip's pieces between the
//! restart index, each piece's triangles in Vulkan's strip order with the parity counted from the
//! piece's start, or its segments, in order; Apple's line loop, fan, quads and quad strip as the
//! Metal probes on the M3 Max drew them ([`RewriteProgram::list`], checked against the probes'
//! lists); positions past the buffer's whole indices read 0, restart included; the draw's
//! arguments carry Metal's instances and bases; a draw without indices reads its positions as
//! its indices; a pooled draw reserves whole words from the pool's counter, and a draw its output
//! cannot hold draws nothing and sets `INDEX_POOL_OVERFLOW`; each primitive is written in the
//! pass's order, Metal's at either provoking vertex mode, or with Metal's last vertex first.

use super::*;
use crate::bind::status::{ExecutionStatus, StatusReport};
use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
use crate::spirv::testing::{assert_valid, committed_profiles};

/// The pool's index bytes in these tests.
const POOL_BYTES: u32 = 4096;
/// The storage buffer offset alignment of these tests.
const ALIGNMENT: u64 = 16;

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

fn bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// Index buffer bytes of `indices`, 4 bytes each where `wide`, else 2.
fn index_bytes(indices: &[u32], wide: bool) -> Vec<u8> {
    indices
        .iter()
        .flat_map(|&i| {
            if wide {
                i.to_le_bytes().to_vec()
            } else {
                (i as u16).to_le_bytes().to_vec()
            }
        })
        .collect()
}

/// Metal's primitives of `program` over `values` (each position's index, as the fetch reads it)
/// restarting at `restart` where the program restarts, in Metal's order at provoking vertex mode
/// 0: [`RewriteProgram::list`], which the probes' lists check below.
fn metal(program: RewriteProgram, values: &[u32], restart: u32) -> Vec<u32> {
    program.list(values, restart, RewriteOrder::First)
}

/// Each position's index: position `start + q` at its index where it lies among the buffer's
/// `indices`, else 0.
fn fetched(indices: &[u32], start: u32, count: u32) -> Vec<u32> {
    (0..u64::from(count))
        .map(|q| {
            let p = u64::from(start) + q;
            indices.get(p as usize).copied().unwrap_or(0)
        })
        .collect()
}

/// One draw.
#[derive(Clone)]
struct Draw {
    program: RewriteProgram,
    wide: bool,
    /// The index buffer's indices from its byte `offset`, after `offset` bytes of 0xAB.
    indices: Vec<u32>,
    offset: u64,
    restart: u32,
    /// Metal's count, instances, first index, base vertex and base instance.
    arguments: [u32; 5],
    /// Indirect: the arguments' offset in the indirect buffer; direct: `None`.
    indirect: Option<u64>,
    /// Pooled: the pool's counter before the pass; owned: `None`.
    counter: Option<u32>,
    /// The workgroups of the count and write dispatches, where not the pass's own.
    groups: Option<u32>,
    /// An indirect draw without indices, its arguments the first four (count, instances, vertex
    /// start, base instance).
    vertices: bool,
    /// The order each primitive is written in.
    order: RewriteOrder,
    /// The list in the private list form (`IndexRewritePass::listed`).
    listed: bool,
}

/// What a pass left: the words, the output's indices (from the pool's index region where
/// pooled), the pool's counter, and the status report.
struct Ran {
    words: Vec<u32>,
    output: Vec<u32>,
    counter: Option<u32>,
    report: StatusReport,
}

impl Draw {
    fn strip(program: RewriteProgram, wide: bool, indices: &[u32], restart: u32) -> Draw {
        Draw {
            program,
            wide,
            indices: indices.to_vec(),
            offset: 0,
            restart,
            arguments: [indices.len() as u32, 1, 0, 0, 0],
            indirect: None,
            counter: None,
            groups: None,
            vertices: false,
            order: RewriteOrder::First,
            listed: false,
        }
    }

    /// An indirect draw without indices of `program`: `count` vertices from `start`.
    fn vertices(
        program: RewriteProgram,
        [count, instances, start, base_instance]: [u32; 4],
    ) -> Draw {
        Draw {
            arguments: [count, instances, start, base_instance, 0xEEEE],
            indirect: Some(8),
            counter: Some(12),
            vertices: true,
            ..Draw::strip(program, true, &[], 0xABCD)
        }
    }

    fn size(&self) -> u64 {
        if self.wide {
            4
        } else {
            2
        }
    }

    /// The pass's run: the indices bound as its range names, the arguments, and the output.
    fn run(&self) -> Ran {
        let size = self.size();
        let mut buffer = vec![0xAB; self.offset as usize];
        buffer.extend(index_bytes(&self.indices, self.wide));
        let end = buffer.len() as u64;
        // The `VkBuffer` runs on to a multiple of 16 bytes (the padding), of bytes no position may
        // read as an index.
        buffer.resize(buffer.len().next_multiple_of(16).max(16), 0xCC);
        let [count, instances, _, base, base_instance] = self.arguments;
        let source = match self.indirect {
            None => RewriteSource::Direct {
                count,
                instances,
                base_vertex: base as i32,
                base_instance,
            },
            Some(at) if self.vertices => RewriteSource::IndirectVertices {
                arguments: IndirectRef::new(0, at),
            },
            Some(at) => RewriteSource::Indirect {
                arguments: IndirectRef::new(0, at),
            },
        };
        let capacity = self.program.capacity(u64::from(count)) as u32 + u32::from(self.listed);
        let output = match self.counter {
            None => RewriteOutput::Owned { capacity },
            Some(_) => RewriteOutput::Pooled {
                pool_bytes: POOL_BYTES,
            },
        };
        let terms = match self.vertices {
            true => (4, 0, 0),
            false => (size, self.offset, end),
        };
        let pass = IndexRewritePass::new(source, self.restart, terms, output, ALIGNMENT, u32::MAX)
            .expect("pass")
            .with_order(self.order);
        let mut pass = match self.listed {
            true => pass.listed(),
            false => pass,
        };
        if let Some(groups) = self.groups {
            pass.groups = groups;
            pass.parameters[Parameter::Groups as usize] = groups;
        }
        let mut rewrite = pass.parameters.to_vec();
        rewrite.resize((IndexRewritePass::BYTES / 4) as usize, 0xEEEE_EEEE);
        // The indirect buffer: the arguments at their offset, bound as the pass names.
        let arguments = match (self.indirect, pass.arguments_range) {
            (Some(at), Some(range)) => {
                let mut b = vec![0xCDu8; (at + 20) as usize];
                b[at as usize..at as usize + 20].copy_from_slice(&bytes(&self.arguments));
                b[range.offset as usize..(range.offset + range.range) as usize].to_vec()
            }
            (None, None) => vec![0xCC; 4],
            _ => panic!("an indirect draw's range"),
        };
        // Exactly the range the pass names, as Vulkan binds it: a shader reads the words that lie
        // wholly inside it.
        let indices = match pass.indices_range {
            Some(range) => {
                buffer[range.offset as usize..(range.offset + range.range) as usize].to_vec()
            }
            None => vec![0xCC; 4],
        };
        let header = (IndexRewritePass::POOL_HEADER_BYTES / 4) as usize;
        let out = match self.counter {
            None => vec![0xDDDD_DDDD; capacity.max(1) as usize],
            Some(counter) => {
                let mut pool = vec![0xDDDD_DDDD; header + (POOL_BYTES / 4) as usize];
                pool[0] = counter;
                pool
            }
        };
        let modules = RewritePhase::ALL.map(|phase| {
            (
                phase,
                rewrite_typed_module(self.program, phase).expect("module"),
            )
        });
        let mut ran = Vec::new();
        for order in [Order::Forward, Order::Reverse] {
            let mut buffers = vec![
                bytes(&rewrite),
                arguments.clone(),
                indices.clone(),
                bytes(&out),
                ExecutionStatus::initial().to_vec(),
            ];
            // The three dispatches in order, each over the buffers the one before left.
            for (phase, module) in &modules {
                let slots = [WORDS, ARGUMENTS, INDICES, OUTPUT, STATUS];
                let res = Resources {
                    buffers: slots
                        .into_iter()
                        .zip(buffers)
                        .map(|(slot, bytes)| Buffer {
                            slot: Some(slot),
                            address: None,
                            bytes,
                        })
                        .collect(),
                    push: Vec::new(),
                    spec: Default::default(),
                };
                let cfg = Config {
                    order,
                    workgroups: [phase.workgroups(&pass), 1, 1],
                    ..Config::default()
                };
                buffers = run(module, &cfg, res).expect("rewrite run").buffers;
            }
            ran.push(buffers);
        }
        assert_eq!(ran[0], ran[1], "order independent");
        let after = &ran[0];
        let status: [u8; ExecutionStatus::BYTES] = after[4].as_slice().try_into().expect("status");
        let out = words(&after[3]);
        let (output, counter) = match self.counter {
            None => (out, None),
            Some(_) => (out[header..].to_vec(), Some(out[0])),
        };
        Ran {
            words: words(&after[0]),
            output,
            counter,
            report: ExecutionStatus::decode(&status),
        }
    }

    /// The indices the draw draws: its command's from the output, every one inside it.
    fn drawn(&self) -> (Vec<u32>, [u32; 5], Ran) {
        let r = self.run();
        let at = (IndexRewritePass::DRAW_OFFSET / 4) as usize;
        let command: [u32; 5] = r.words[at..at + 5].try_into().expect("command");
        let first = command[2] as usize;
        let n = command[0] as usize;
        assert!(
            first + n <= r.output.len(),
            "the draw reads past its output"
        );
        let list = r.output[first..first + n].to_vec();
        (list, command, r)
    }
}

#[test]
fn every_program_is_valid_on_every_committed_profile() {
    for (_, p) in committed_profiles().expect("profiles") {
        for program in RewriteProgram::ALL {
            for phase in RewritePhase::ALL {
                assert_valid(rewrite_typed_module(program, phase).expect("module"), &p);
            }
        }
    }
}

/// The pass binds the arguments and the whole indices from aligned offsets, and states the
/// residuals, the whole indices, the index size, `k'` and the output.
#[test]
fn the_pass_binds_aligned_ranges_and_states_its_parameters() {
    let pass = IndexRewritePass::new(
        RewriteSource::Indirect {
            arguments: IndirectRef::new(0, 68),
        },
        3,
        (4, 8, 30),
        RewriteOutput::Pooled { pool_bytes: 64 },
        64,
        1 << 27,
    )
    .expect("pass");
    assert_eq!(
        pass.arguments_range,
        Some(BufferRange {
            offset: 64,
            range: 24
        })
    );
    assert_eq!(
        pass.indices_range,
        Some(BufferRange {
            offset: 0,
            range: 28
        })
    );
    assert_eq!(pass.parameters[..8], [2, 8, 5, 1, 3, 64, 1, 1]);
    // The indices' range ends at a whole word: nine 16-bit indices end at byte 18, and ten from
    // byte 2 at byte 22.
    let indices_range = |offset, end| {
        IndexRewritePass::new(
            RewriteSource::Indirect {
                arguments: IndirectRef::new(0, 0),
            },
            3,
            (2, offset, end),
            RewriteOutput::Pooled { pool_bytes: 64 },
            64,
            1 << 27,
        )
        .expect("pass")
        .indices_range
    };
    for (offset, end, range) in [(0, 18, 20), (2, 22, 24), (2, 20, 20), (66, 82, 20)] {
        assert_eq!(
            indices_range(offset, end),
            Some(BufferRange {
                offset: offset - offset % 64,
                range
            }),
            "from {offset} to {end}"
        );
    }
    assert_eq!(pass.groups, 1);
    assert_eq!(pass.parameters[8..], [0; 8]);
    let direct = IndexRewritePass::new(
        RewriteSource::Direct {
            count: 7,
            instances: 2,
            base_vertex: -1,
            base_instance: 5,
        },
        0xFFFF,
        (2, 30, 30),
        RewriteOutput::Owned { capacity: 15 },
        64,
        1 << 27,
    )
    .expect("direct");
    assert_eq!(direct.arguments_range, None);
    assert_eq!(direct.indices_range, None);
    assert_eq!(direct.parameters[..8], [0, 0, 0, 0, 0xFFFF, 15, 0, 1]);
    assert_eq!(direct.parameters[8..13], [7, 2, 0, u32::MAX, 5]);
    let refused = |source, size, offset, output| {
        IndexRewritePass::new(source, 0, (size, offset, 64), output, 64, 1 << 27).is_none()
    };
    let owned = RewriteOutput::Owned { capacity: 1 };
    let at = |offset| RewriteSource::Indirect {
        arguments: IndirectRef::new(0, offset),
    };
    assert!(refused(at(2), 4, 0, owned), "arguments' offset");
    assert!(refused(at(0), 3, 0, owned), "index size");
    assert!(refused(at(0), 4, 2, owned), "index offset");
    assert!(refused(
        at(0),
        4,
        0,
        RewriteOutput::Pooled { pool_bytes: 6 }
    ));
}

/// Probe 5's strips, drawn directly: each restarts at `k'` alone, the all-ones index drawn as a
/// vertex where it is not `k'`; 16- and 32-bit alike.
#[test]
fn a_strip_restarts_at_its_index_alone() {
    for program in RewriteProgram::ALL {
        for (wide, indices, restart) in [
            (false, vec![1, 2, 3, 0, 4, 5, 6, 0xFFFF, 7, 8, 9], 0),
            (false, vec![0, 1, 2, 3, 4, 5, 6, 0xFFFF, 7, 8, 9], 3),
            (false, vec![1, 2, 3, 0, 4, 5, 6, 0xFFFF, 7, 8, 9], 0xFFFF),
            (
                true,
                vec![1, 2, 3, 0, 4, 5, 6, u32::MAX, 7, 8, 0xFFFF, 9, 10],
                0,
            ),
            (
                true,
                vec![0, 1, 2, 3, 4, 5, 6, u32::MAX, 7, 8, 0xFFFF, 9, 10],
                0xFFFF,
            ),
            (true, vec![3, 3, 0, 1, 3, 2, 4, 5, 3], 3),
            (true, vec![3, 3, 3], 3),
            (true, vec![1, 2], 3),
            (true, vec![], 3),
        ] {
            let draw = Draw::strip(program, wide, &indices, restart);
            let (list, command, r) = draw.drawn();
            let expected = metal(program, &indices, restart);
            assert_eq!(
                list, expected,
                "{program:?} {wide} {indices:?} k' {restart}"
            );
            assert_eq!(command[1..], [1, 0, 0, 0], "{indices:?}");
            assert_eq!(
                r.words[Output::Primitives as usize] as usize,
                expected.len() / program.vertices() as usize
            );
            assert!(!r.report.index_pool_overflow);
        }
    }
}

/// A strip longer than a workgroup's invocations, over one workgroup and over several: its runs'
/// pieces cross between runs and between groups, and each run's first triangles take their
/// piece's parity from the runs before, its group's and the groups' before it. The other programs
/// draw the same indices over three groups: the fan's first index, and the loop's closing
/// segment, reach across them.
#[test]
fn a_strip_of_many_runs_keeps_each_piece_s_parity() {
    // A deterministic mix: restarts at 0 here and there, long pieces and short ones.
    let mut state = 0x1234_5678u32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    for (len, rate) in [(600, 7), (1000, 97), (777, 3), (1300, 1000)] {
        let indices: Vec<u32> = (0..len)
            .map(|_| {
                let r = next();
                if r % rate == 0 {
                    0
                } else {
                    1 + r % 50
                }
            })
            .collect();
        for program in RewriteProgram::ALL {
            // The strips each way; the other programs (the quad strip has a test of its own)
            // over three groups.
            let strip = matches!(
                program,
                RewriteProgram::LineStrip | RewriteProgram::TriangleStrip
            );
            let ways: &[(bool, Option<u32>)] = match strip {
                true => &[
                    (false, None),
                    (true, None),
                    (false, Some(3)),
                    (true, Some(2)),
                ],
                false => &[(false, Some(3))],
            };
            for &(wide, groups) in ways {
                let draw = Draw {
                    groups,
                    ..Draw::strip(program, wide, &indices, 0)
                };
                let (list, _, _) = draw.drawn();
                assert_eq!(
                    list,
                    metal(program, &indices, 0),
                    "{program:?} {wide} {len} {rate} {groups:?}"
                );
            }
        }
    }
}

/// The count and write dispatches take a workgroup per 4096 positions the walk may take, from 1
/// to 256: a direct draw's count bounds the walk, an indirect draw's by the buffer's whole
/// indices and the most positions a step spans past them, and an indirect draw without indices
/// takes all 256.
#[test]
fn the_groups_follow_the_walk() {
    let groups = |source, inside: u64| {
        let output = RewriteOutput::Pooled { pool_bytes: 64 };
        IndexRewritePass::new(source, 0, (4, 0, inside * 4), output, 64, u32::MAX)
            .expect("pass")
            .groups
    };
    let direct = |count| RewriteSource::Direct {
        count,
        instances: 1,
        base_vertex: 0,
        base_instance: 0,
    };
    let indirect = RewriteSource::Indirect {
        arguments: IndirectRef::new(0, 0),
    };
    assert_eq!(groups(direct(10), 10), 1);
    assert_eq!(groups(direct(4096), 1 << 20), 1);
    assert_eq!(groups(direct(4097), 1 << 20), 2);
    assert_eq!(groups(direct(1_000_000), 1_000_000), 245);
    assert_eq!(groups(direct(1_000_000), 0), 1);
    assert_eq!(groups(direct(u32::MAX), 1 << 29), 256);
    assert_eq!(groups(indirect, 4092), 1);
    assert_eq!(groups(indirect, 4093), 2);
    assert_eq!(groups(indirect, 2_000_000), 256);
    let vertices = RewriteSource::IndirectVertices {
        arguments: IndirectRef::new(0, 0),
    };
    assert_eq!(groups(vertices, 0), 256);
}

/// An indirect draw reads Metal's arguments (its first index, instances and bases), reserves
/// whole words from the pool's counter, and draws from there; a draw the pool cannot hold draws
/// nothing and sets the overflow bit.
#[test]
fn an_indirect_draw_reserves_its_list_in_the_pool() {
    let indices: Vec<u32> = vec![9, 9, 1, 2, 3, 0, 4, 5, 6, 7, 0, 8];
    for program in RewriteProgram::ALL {
        let draw = Draw {
            arguments: [8, 3, 2, 0xFFFF_FFFE, 4],
            indirect: Some(36),
            counter: Some(40),
            ..Draw::strip(program, false, &indices, 0)
        };
        let (list, command, r) = draw.drawn();
        let expected = metal(program, &fetched(&indices, 2, 8), 0);
        assert_eq!(list, expected, "{program:?}");
        assert_eq!(command[1..], [3, 10, 0xFFFF_FFFE, 4], "{program:?}");
        assert_eq!(r.counter, Some(40 + 4 * expected.len() as u32));
        assert!(!r.report.index_pool_overflow);
        // No room: nothing drawn, the counter kept, the bit set.
        let full = Draw {
            counter: Some(POOL_BYTES - 4),
            ..draw.clone()
        };
        let (list, command, r) = full.drawn();
        assert!(list.is_empty());
        assert_eq!(command[0], 0);
        assert_eq!(r.counter, Some(POOL_BYTES - 4));
        assert!(r.report.index_pool_overflow);
    }
}

/// Positions past the buffer's whole indices read 0: they restart where `k'` is 0, and are
/// vertex 0 otherwise, the primitives past the walk written without one; an index offset inside
/// a word, and a partial last index, read as Metal's fetch does.
#[test]
fn positions_past_the_buffer_read_index_0() {
    let indices = vec![5, 6, 7, 8, 9];
    for program in RewriteProgram::ALL {
        for (restart, count) in [(0, 9), (3, 9), (3, 40), (0, 40), (3, 2)] {
            for wide in [false, true] {
                let draw = Draw {
                    arguments: [count, 1, 0, 0, 0],
                    offset: if wide { 4 } else { 6 },
                    ..Draw::strip(program, wide, &indices, restart)
                };
                let (list, _, _) = draw.drawn();
                let expected = metal(program, &fetched(&indices, 0, count), restart);
                assert_eq!(list, expected, "{program:?} {wide} k' {restart} {count}");
            }
        }
        // An indirect draw from past the buffer reads 0 throughout.
        let draw = Draw {
            arguments: [6, 1, 9, 0, 0],
            indirect: Some(0),
            counter: Some(0),
            ..Draw::strip(program, true, &indices, 4)
        };
        let (list, _, _) = draw.drawn();
        assert_eq!(
            list,
            metal(program, &[0; 6], 4),
            "{program:?} from past the buffer"
        );
    }
}

/// The last whole index of a word the buffer's indices end inside is read, not 0: an odd count
/// of 16-bit indices from a multiple of 4 bytes, or an even count from 2 past one, direct and
/// indirect (the consumer's nine 16-bit indices restarting at 3, from an 18- or a 20-byte buffer,
/// drew their last triangle to vertex 0 on an M3 Max where ten were exact).
#[test]
fn the_last_index_of_a_partial_word_is_read() {
    let nine: Vec<u32> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8];
    let ten: Vec<u32> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    let seven: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7];
    for program in RewriteProgram::ALL {
        for (indices, offset) in [
            (&nine, 0),
            (&nine, 2),
            (&ten, 2),
            (&ten, 0),
            (&seven, 4),
            (&seven, 6),
        ] {
            let count = indices.len() as u32;
            let direct = Draw {
                offset,
                ..Draw::strip(program, false, indices, 3)
            };
            let indirect = Draw {
                arguments: [count, 1, 0, 0, 0],
                indirect: Some(4),
                counter: Some(0),
                ..direct.clone()
            };
            for (draw, how) in [(direct, "direct"), (indirect, "indirect")] {
                let (list, _, _) = draw.drawn();
                assert_eq!(
                    list,
                    metal(program, indices, 3),
                    "{program:?} {how}: {count} from byte {offset}"
                );
            }
        }
    }
}

/// The lists of Apple's private types are those the Metal probes drew on the M3 Max (macOS
/// 27.0.1, `primtypes` probe: each primitive's vertices in barycentric order, its flat vertex the
/// first at provoking vertex mode 0 and the last at mode 1), and a draw without indices draws the
/// list of its positions.
#[test]
fn the_private_lists_are_the_probes() {
    use RewriteProgram::{LineLoop, QuadStrip, Quads, TriangleFan};
    let drawn: &[(RewriteProgram, u32, &[u32])] = &[
        (LineLoop, 0, &[]),
        (LineLoop, 1, &[]),
        (LineLoop, 2, &[0, 1, 1, 0]),
        (LineLoop, 5, &[0, 1, 1, 2, 2, 3, 3, 4, 4, 0]),
        (TriangleFan, 2, &[]),
        (TriangleFan, 3, &[0, 1, 2]),
        (TriangleFan, 6, &[0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 5]),
        (Quads, 3, &[]),
        (Quads, 4, &[0, 1, 3, 1, 2, 3]),
        (Quads, 7, &[0, 1, 3, 1, 2, 3]),
        (Quads, 9, &[0, 1, 3, 1, 2, 3, 4, 5, 7, 5, 6, 7]),
        (QuadStrip, 3, &[]),
        (QuadStrip, 4, &[0, 1, 3, 2, 0, 3]),
        (QuadStrip, 5, &[0, 1, 3, 2, 0, 3]),
        (
            QuadStrip,
            8,
            &[0, 1, 3, 2, 0, 3, 2, 3, 5, 4, 2, 5, 4, 5, 7, 6, 4, 7],
        ),
    ];
    for &(program, count, list) in drawn {
        let positions: Vec<u32> = (0..count).collect();
        assert_eq!(
            program.list(&positions, u32::MAX, RewriteOrder::First),
            list,
            "{program:?} {count}"
        );
        assert_eq!(
            program.sequential_list(count, RewriteOrder::First),
            list,
            "{program:?} {count}"
        );
        assert!(list.len() as u64 <= program.capacity(u64::from(count)));
    }
    // Every count from 0 to 40: the CPU's list of positions is the general one's.
    for program in RewriteProgram::ALL {
        for count in 0..40 {
            let positions: Vec<u32> = (0..count).collect();
            let list = program.sequential_list(count, RewriteOrder::First);
            assert_eq!(
                list,
                program.list(&positions, u32::MAX, RewriteOrder::First),
                "{program:?} {count}"
            );
            assert_eq!(list.len() as u64, program.capacity(u64::from(count)));
        }
    }
    // The probes' 16-bit draws of `40 + 2i` with the all-ones index at position 5: the line
    // loop, fan and quads draw it as a vertex; the quad strip restarts there, its second piece
    // pairing from its own start.
    let mut values: Vec<u32> = (0..12).map(|i| 40 + 2 * i).collect();
    values[5] = 0xFFFF;
    assert_eq!(
        Quads.list(&values, 0xFFFF, RewriteOrder::First),
        [40, 42, 46, 42, 44, 46, 48, 0xFFFF, 54, 0xFFFF, 52, 54, 56, 58, 62, 58, 60, 62]
    );
    assert_eq!(
        QuadStrip.list(&values, 0xFFFF, RewriteOrder::First),
        [40, 42, 46, 44, 40, 46, 52, 54, 58, 56, 52, 58, 56, 58, 62, 60, 56, 62]
    );
    assert_eq!(
        TriangleFan.list(&values, 0xFFFF, RewriteOrder::First)[9..12],
        [40, 48, 0xFFFF]
    );
    assert_eq!(
        LineLoop.list(&values, 0xFFFF, RewriteOrder::First).len(),
        24
    );
}

/// An indirect draw without indices writes the list of its positions from 0, whatever its count,
/// into the pool, and draws it from its vertex start as base vertex with Metal's instances and
/// base instance; none of its positions restarts. Its pass takes every group, as the GPU alone
/// knows its count (`IndexRewritePass::groups`, which the groups test checks); here one group or
/// three, the interpreter being slow.
#[test]
fn a_draw_without_indices_lists_its_positions() {
    let mut draws = Vec::new();
    for program in RewriteProgram::ALL {
        for (count, groups) in [
            (0, 1),
            (1, 1),
            (2, 3),
            (3, 1),
            (4, 3),
            (5, 1),
            (9, 3),
            (130, 3),
        ] {
            draws.push((program, count, Some(groups)));
        }
    }
    for (program, count, groups) in draws {
        let draw = Draw {
            groups,
            ..Draw::vertices(program, [count, 2, 0xFFFF, 7])
        };
        let (list, command, r) = draw.drawn();
        let expected = program.sequential_list(count, RewriteOrder::First);
        assert_eq!(list, expected, "{program:?} {count}");
        assert_eq!(command[1..], [2, 3, 0xFFFF, 7], "{program:?} {count}");
        assert_eq!(r.counter, Some(12 + 4 * expected.len() as u32));
        assert!(!r.report.index_pool_overflow);
    }
}

/// A quad strip of many runs over several groups, restarting here and there: each piece pairs
/// from its own start, whichever run or group it began in, so a run's first quads take the
/// parity of its piece's positions before it.
#[test]
fn a_quad_strip_pairs_from_each_piece_s_start() {
    let mut state = 0x9E37_79B9u32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    for (len, rate) in [(700, 5), (1500, 61), (999, 2), (2048, 333)] {
        let indices: Vec<u32> = (0..len)
            .map(|_| {
                let r = next();
                if r % rate == 0 {
                    7
                } else {
                    8 + r % 40
                }
            })
            .collect();
        for (groups, wide) in [
            (None, false),
            (Some(1), true),
            (Some(3), false),
            (Some(5), true),
        ] {
            let draw = Draw {
                groups,
                ..Draw::strip(RewriteProgram::QuadStrip, wide, &indices, 7)
            };
            let (list, _, _) = draw.drawn();
            assert_eq!(
                list,
                metal(RewriteProgram::QuadStrip, &indices, 7),
                "{len} {rate} {groups:?} {wide}"
            );
        }
    }
}

/// The orders of a primitive: Metal's at mode 0, at mode 1 (a triangle strip's odd triangle
/// `(i + 1, i, i + 2)`, its even one and every other primitive as at mode 0), and Metal's at mode
/// 1 with the last vertex first, so a first-vertex pipeline reads each primitive's flat inputs
/// from the vertex Metal's mode 1 reads them from (Metal probes on the M3 Max, macOS 27.0.1: line
/// list `2j + 1`, line strip `j + 1`, triangle list `3j + 2`, strip `j + 2`, line loop the
/// segment's second, fan `j + 2`, quads and quad strip `d`), each triangle keeping its winding.
#[test]
fn the_orders_are_metal_s_at_either_mode() {
    use RewriteOrder::{First, Last, LastFirst};
    use RewriteProgram::{LineList, LineStrip, TriangleList, TriangleStrip};
    assert_eq!(
        TriangleStrip.sequential_list(5, First),
        [0, 1, 2, 1, 3, 2, 2, 3, 4]
    );
    assert_eq!(
        TriangleStrip.sequential_list(5, Last),
        [0, 1, 2, 2, 1, 3, 2, 3, 4]
    );
    assert_eq!(
        TriangleStrip.sequential_list(5, LastFirst),
        [2, 0, 1, 3, 2, 1, 4, 2, 3]
    );
    assert_eq!(LineStrip.sequential_list(3, LastFirst), [1, 0, 2, 1]);
    assert_eq!(LineList.sequential_list(5, First), [0, 1, 2, 3]);
    assert_eq!(LineList.sequential_list(5, LastFirst), [1, 0, 3, 2]);
    assert_eq!(TriangleList.sequential_list(7, First), [0, 1, 2, 3, 4, 5]);
    assert_eq!(
        TriangleList.sequential_list(7, LastFirst),
        [2, 0, 1, 5, 3, 4]
    );
    // Metal's mode-1 flat vertex of each list primitive, by program, of the positions `0..n`.
    let flat = |program: RewriteProgram, n: u32| -> Vec<u32> {
        let list = program.sequential_list(n, Last);
        let v = program.vertices() as usize;
        list.chunks(v).map(|p| p[v - 1]).collect()
    };
    assert_eq!(flat(LineList, 6), [1, 3, 5]);
    assert_eq!(flat(LineStrip, 4), [1, 2, 3]);
    assert_eq!(flat(TriangleList, 6), [2, 5]);
    assert_eq!(flat(TriangleStrip, 6), [2, 3, 4, 5]);
    assert_eq!(flat(RewriteProgram::LineLoop, 4), [1, 2, 3, 0]);
    assert_eq!(flat(RewriteProgram::TriangleFan, 5), [2, 3, 4]);
    assert_eq!(flat(RewriteProgram::Quads, 4), [3, 3]);
    assert_eq!(flat(RewriteProgram::QuadStrip, 6), [3, 3, 5, 5]);
    // The winding: a signed area of positions on a convex curve, so each position's point is
    // distinct and no three are collinear.
    let point = |i: u32| {
        let t = f64::from(i) * 0.37;
        (t.cos() * (1.0 + 0.1 * t), t.sin() * (1.0 + 0.1 * t))
    };
    let area = |t: &[u32]| {
        let (a, b, c) = (point(t[0]), point(t[1]), point(t[2]));
        (b.0 - a.0) * (c.1 - a.1) - (c.0 - a.0) * (b.1 - a.1)
    };
    for program in RewriteProgram::ALL {
        for n in 0..24 {
            let first = program.sequential_list(n, First);
            let last = program.sequential_list(n, Last);
            let led = program.sequential_list(n, LastFirst);
            let v = program.vertices() as usize;
            assert_eq!(first.len(), last.len(), "{program:?} {n}");
            assert_eq!(first.len(), led.len(), "{program:?} {n}");
            for ((f, l), r) in first.chunks(v).zip(last.chunks(v)).zip(led.chunks(v)) {
                // The last-first order leads with Metal's mode-1 flat vertex, then the rest in
                // their order.
                assert_eq!(r[0], l[v - 1], "{program:?} {n}");
                assert_eq!(r[1..], l[..v - 1], "{program:?} {n}");
                if v == 3 {
                    assert!(area(f) * area(l) > 0.0, "{program:?} {n} {f:?} {l:?}");
                    assert!(area(f) * area(r) > 0.0, "{program:?} {n} {f:?} {r:?}");
                } else if program != TriangleStrip {
                    assert_eq!(f, l, "{program:?} {n}");
                }
            }
            if program != TriangleStrip {
                assert_eq!(first, last, "{program:?} {n}");
            }
        }
    }
}

/// The pass writes each order as the CPU's list does, for every program: restarting draws,
/// a fan's first index and a loop's closing segment over groups, the primitives past the walk,
/// and an indirect draw without indices.
#[test]
fn the_pass_writes_each_order() {
    let mut state = 0x0BAD_5EEDu32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let long: Vec<u32> = (0..700)
        .map(|_| {
            let r = next();
            if r % 9 == 0 {
                0
            } else {
                1 + r % 50
            }
        })
        .collect();
    let short = vec![1, 2, 3, 0, 4, 5, 6, 0xFFFF, 7, 8, 9];
    for order in [RewriteOrder::Last, RewriteOrder::LastFirst] {
        for program in RewriteProgram::ALL {
            let draws = [
                Draw::strip(program, false, &short, 0),
                Draw {
                    groups: Some(3),
                    ..Draw::strip(program, true, &long, 0)
                },
                // Nine positions past five indices: the primitives past the walk.
                Draw {
                    arguments: [40, 1, 0, 0, 0],
                    ..Draw::strip(program, true, &[5, 6, 7, 8, 9], 3)
                },
            ];
            for draw in draws {
                let draw = Draw { order, ..draw };
                let (list, _, _) = draw.drawn();
                let count = draw.arguments[0];
                let values = fetched(&draw.indices, 0, count);
                assert_eq!(
                    list,
                    program.list(&values, draw.restart, order),
                    "{program:?} {order:?} {count}"
                );
            }
            let draw = Draw {
                groups: Some(3),
                order,
                ..Draw::vertices(program, [130, 1, 4, 0])
            };
            let (list, _, _) = draw.drawn();
            assert_eq!(
                list,
                program.sequential_list(130, order),
                "{program:?} {order:?} without indices"
            );
        }
    }
}

/// In the private list form the pass writes the same list one word on, Metal's base vertex (a
/// direct or indexed indirect draw's base vertex, an indirect draw's vertex start) in the word
/// before it, and the draw as a `VkDrawIndirectCommand` of the list from its first index as
/// first vertex, with Metal's instances and base instance; the reservation takes the word too,
/// and a list the room holds but for the word draws nothing and says so.
#[test]
fn the_private_list_form_writes_the_base_vertex_before_its_list() {
    let indices: Vec<u32> = vec![9, 9, 1, 2, 3, 0, 4, 5, 6, 7, 0, 8];
    for program in RewriteProgram::ALL {
        let direct = Draw {
            arguments: [9, 2, 0, 0x8000_0005, 3],
            ..Draw::strip(program, true, &indices, 0)
        };
        let indirect = Draw {
            arguments: [8, 3, 2, 0xFFFF_FFFE, 4],
            indirect: Some(36),
            counter: Some(40),
            ..Draw::strip(program, false, &indices, 0)
        };
        let vertices = Draw {
            groups: Some(1),
            ..Draw::vertices(program, [7, 2, 0x8000_0010, 5])
        };
        for (draw, base, instances, base_instance) in [
            (direct, 0x8000_0005, 2, 3),
            (indirect, 0xFFFF_FFFE, 3, 4),
            (vertices, 0x8000_0010, 2, 5),
        ] {
            let (plain, _, _) = draw.drawn();
            let listed = Draw {
                listed: true,
                ..draw.clone()
            };
            let (list, command, r) = listed.drawn();
            assert_eq!(list, plain, "{program:?}");
            let first = command[2] as usize;
            let start = match draw.counter {
                Some(counter) => (counter / 4) as usize + 1,
                None => 1,
            };
            assert_eq!(first, start, "{program:?}: the list's first index");
            assert_eq!(r.output[first - 1], base, "{program:?}: the base vertex");
            assert_eq!(
                command,
                [
                    plain.len() as u32,
                    instances,
                    first as u32,
                    base_instance,
                    0
                ],
                "{program:?}"
            );
            if let (Some(before), Some(after)) = (draw.counter, r.counter) {
                assert_eq!(after, before + 4 * (plain.len() as u32 + 1), "{program:?}");
            }
            assert!(!r.report.index_pool_overflow);
        }
        // Room for the list but not its base vertex's word: nothing drawn, the bit set.
        let needed = metal(program, &fetched(&indices, 2, 8), 0).len() as u32;
        let tight = Draw {
            arguments: [8, 3, 2, 7, 4],
            indirect: Some(36),
            counter: Some(POOL_BYTES - 4 * needed),
            listed: true,
            ..Draw::strip(program, false, &indices, 0)
        };
        let (list, command, r) = tight.drawn();
        if needed > 0 {
            assert!(list.is_empty(), "{program:?}");
            assert_eq!(command[0], 0, "{program:?}");
            assert!(r.report.index_pool_overflow, "{program:?}");
        }
    }
}
