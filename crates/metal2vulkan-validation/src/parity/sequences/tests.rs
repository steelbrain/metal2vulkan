//! The sequences' CPU tests: the generator is deterministic and makes race-free sequences that
//! reach every command; the order finds the races it should; the model draws and mixes as the
//! fixture does; the minimizer keeps a failure in a race-free sequence; and the committed records
//! are current and agree with the model, each seed answered or a recorded Metal-side gap.

use super::super::record::{read_lines, stale, Refusal, Value};
use super::super::records_path;
use super::generate::generate;
use super::hazards::{
    parse, verdict, vulkan_commands, Call, Encoded, Handles, Hazard, MetalOrder, Named, Verdict,
};
use super::minimize::{minimize, removals, tidy};
use super::model::{byte_differences, mix, records as model_records, run};
use super::order::{accesses, problems, Order};
use super::program::{
    case_name, At, BufferSpec, Channel, Command, CommandBufferSpec, Encoder, Format, HeapSpec,
    Kernel, Kind, Load, Place, Program, Role, Step, Surface, TextureSpec, ViewForm, ViewSpec,
    BUFFER_BYTES, HEAP_WINDOW, ROW_BYTES, TEXEL_BYTES, TEXTURE_BYTES,
};
use super::{fingerprint, metal_gap, METAL_GAP, NAME, SEEDS};
use std::collections::{BTreeMap, BTreeSet};

/// The seeds the generator is checked over beyond those recorded.
const MORE_SEEDS: u64 = 448;

#[test]
fn generation_is_deterministic_and_race_free_and_the_model_runs() {
    for seed in 0..SEEDS + MORE_SEEDS {
        let program = generate(seed);
        let again = generate(seed);
        assert_eq!(format!("{program:?}"), format!("{again:?}"), "seed {seed}");
        let problems = problems(&program);
        assert!(
            problems.is_empty(),
            "seed {seed}:\n{}\n{program}",
            problems.join("\n")
        );
        let state = run(&program);
        assert_eq!(state.buffers.len(), program.buffers.len());
        assert_eq!(state.textures.len(), program.textures.len());
    }
}

/// The recorded seeds reach every command and every kind of synchronization the generator makes.
#[test]
fn the_seeds_reach_every_command() {
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    let mut count = |what: &'static str| *counts.entry(what).or_default() += 1;
    for seed in 0..SEEDS {
        let program = generate(seed);
        for heap in &program.heaps {
            count(if heap.private {
                "private heap"
            } else {
                "shared heap"
            });
            count(if heap.tracked {
                "tracked heap"
            } else {
                "untracked heap"
            });
        }
        let mut placed = std::collections::BTreeSet::new();
        for buffer in &program.buffers {
            if buffer.private {
                count("private buffer");
            }
            if !buffer.tracked {
                count("untracked buffer");
            }
            match buffer.place {
                Place::Own => {}
                Place::NoCopy => count("buffer over a host page"),
                Place::Heap { heap, offset } => {
                    count("buffer in a heap");
                    let step = super::program::HEAP_STEP;
                    if placed.contains(&(heap, offset))
                        || placed.contains(&(heap, offset + step))
                        || offset >= step && placed.contains(&(heap, offset - step))
                    {
                        count("overlapping heap buffers");
                    }
                    placed.insert((heap, offset));
                }
            }
        }
        for texture in &program.textures {
            count(if texture.tracked {
                "tracked texture"
            } else {
                "untracked texture"
            });
            if texture.slices > 1 {
                count("texture of two slices");
            }
            if texture.heap {
                count("texture in a heap");
            }
        }
        for view in &program.views {
            count(match view.form {
                ViewForm::Whole => "view of a whole texture",
                ViewForm::Slices => "view of a slice",
                ViewForm::Swizzled(_) => "swizzled view",
            });
            if view.format == Format::Bgra8Unorm {
                count("view in another format");
            }
        }
        for spec in &program.command_buffers {
            if spec.queue > 0 {
                count("second or third queue");
            }
            for step in &spec.steps {
                match step {
                    Step::Wait(_) => count("event wait"),
                    Step::Signal(_) => count("event signal"),
                    Step::Encoder(encoder) => {
                        if !encoder.waits.is_empty() {
                            count("fence wait");
                        }
                        count(match encoder.kind {
                            Kind::Blit => "blit encoder",
                            Kind::Compute { concurrent: false } => "serial compute encoder",
                            Kind::Compute { concurrent: true } => "concurrent compute encoder",
                            Kind::Render {
                                load: Load::Clear(_),
                                ..
                            } => "render pass clearing",
                            Kind::Render {
                                load: Load::Load, ..
                            } => "render pass loading",
                        });
                        match encoder.kind {
                            Kind::Render {
                                target: Surface::View(_),
                                ..
                            } => count("render pass through a view"),
                            Kind::Render { slice: 1, .. } => count("render pass of slice 1"),
                            _ => {}
                        }
                        for command in &encoder.commands {
                            count(match command {
                                Command::Fill { .. } => "fill",
                                Command::Copy { .. } => "copy",
                                Command::ToTexture {
                                    to: Surface::View(_),
                                    ..
                                } => "copy to a view",
                                Command::FromTexture {
                                    from: Surface::View(_),
                                    ..
                                } => "copy from a view",
                                Command::ToTexture { .. } => "copy to a texture",
                                Command::FromTexture { .. } => "copy from a texture",
                                Command::TexelRead {
                                    from: Surface::View(_),
                                    ..
                                } => "texel read through a view",
                                Command::TexelWrite {
                                    to: Surface::View(_),
                                    ..
                                } => "texel write through a view",
                                Command::TexelRead { .. } => "texel read",
                                Command::TexelWrite { .. } => "texel write",
                                Command::Dispatch {
                                    indirect: Some(_), ..
                                } => "indirect dispatch",
                                Command::Dispatch {
                                    kernel: Kernel::Copy,
                                    ..
                                } => "dispatch of seq_copy",
                                Command::Dispatch {
                                    kernel: Kernel::Mix,
                                    ..
                                } => "dispatch of seq_mix",
                                Command::Barrier { textures: false } => "memory barrier",
                                Command::Barrier { textures: true } => "memory barrier of textures",
                                Command::Draw {
                                    indirect: Some(_), ..
                                } => "indirect draw",
                                Command::Draw { .. } => "draw",
                            });
                        }
                    }
                }
            }
        }
    }
    let wanted = [
        "private heap",
        "shared heap",
        "tracked heap",
        "untracked heap",
        "buffer in a heap",
        "overlapping heap buffers",
        "buffer over a host page",
        "texture of two slices",
        "texture in a heap",
        "view of a whole texture",
        "view of a slice",
        "swizzled view",
        "view in another format",
        "render pass through a view",
        "render pass of slice 1",
        "copy to a view",
        "copy from a view",
        "texel read",
        "texel write",
        "texel read through a view",
        "texel write through a view",
        "memory barrier of textures",
        "private buffer",
        "untracked buffer",
        "tracked texture",
        "untracked texture",
        "second or third queue",
        "event wait",
        "event signal",
        "fence wait",
        "blit encoder",
        "serial compute encoder",
        "concurrent compute encoder",
        "render pass clearing",
        "render pass loading",
        "fill",
        "copy",
        "copy to a texture",
        "copy from a texture",
        "indirect dispatch",
        "dispatch of seq_copy",
        "dispatch of seq_mix",
        "memory barrier",
        "indirect draw",
        "draw",
    ];
    eprintln!("sequences: what seeds 0..{SEEDS} reach: {counts:?}");
    let missing: Vec<&str> = wanted
        .into_iter()
        .filter(|what| !counts.contains_key(what))
        .collect();
    assert!(missing.is_empty(), "no seed reaches {missing:?}");
}

fn data(tracked: bool) -> BufferSpec {
    BufferSpec {
        role: Role::Data,
        private: false,
        tracked,
        place: Place::Own,
        initial: vec![0; BUFFER_BYTES as usize],
    }
}

fn encoder(kind: Kind, waits: &[usize], updates: &[usize], commands: Vec<Command>) -> Step {
    Step::Encoder(Encoder {
        kind,
        waits: waits.to_vec(),
        updates: updates.to_vec(),
        commands,
    })
}

fn fill(buffer: usize, start: u64, end: u64) -> Command {
    Command::Fill {
        buffer,
        start,
        end,
        value: 0x5A,
    }
}

fn at(buffer: usize, offset: u64) -> At {
    At { buffer, offset }
}

fn copy_word(from: u64, to: u64) -> Command {
    Command::Dispatch {
        kernel: Kernel::Copy,
        a: at(0, from),
        b: at(0, to),
        indirect: None,
    }
}

/// A sequence of `buffers` and command buffers `(queue, steps)`, its events and fences counted.
fn program(buffers: Vec<BufferSpec>, command_buffers: Vec<(usize, Vec<Step>)>) -> Program {
    let mut events = 0;
    let mut fences = 0;
    for (_, steps) in &command_buffers {
        for step in steps {
            match step {
                Step::Wait(event) | Step::Signal(event) => events = events.max(event + 1),
                Step::Encoder(encoder) => {
                    for &fence in encoder.waits.iter().chain(&encoder.updates) {
                        fences = fences.max(fence + 1);
                    }
                }
            }
        }
    }
    Program {
        heaps: Vec::new(),
        buffers,
        textures: Vec::new(),
        views: Vec::new(),
        queues: 1 + command_buffers
            .iter()
            .map(|(queue, _)| *queue)
            .max()
            .unwrap_or(0),
        events,
        fences,
        command_buffers: command_buffers
            .into_iter()
            .map(|(queue, steps)| CommandBufferSpec { queue, steps })
            .collect(),
    }
}

fn races(program: &Program) -> bool {
    problems(program)
        .iter()
        .any(|problem| problem.starts_with("race"))
}

#[test]
fn the_order_finds_races_inside_an_encoder() {
    // A blit encoder's commands are never taken to be ordered.
    let blit = |tracked| {
        program(
            vec![data(tracked)],
            vec![(
                0,
                vec![encoder(
                    Kind::Blit,
                    &[],
                    &[],
                    vec![fill(0, 0, 8), fill(0, 4, 12)],
                )],
            )],
        )
    };
    assert!(races(&blit(true)) && races(&blit(false)));
    // Disjoint bytes never race.
    let disjoint = program(
        vec![data(false)],
        vec![(
            0,
            vec![encoder(
                Kind::Blit,
                &[],
                &[],
                vec![fill(0, 0, 4), fill(0, 4, 8)],
            )],
        )],
    );
    assert!(problems(&disjoint).is_empty());
    // A serial compute encoder orders a tracked buffer's accesses, and only a tracked one's.
    let compute = |concurrent, tracked, commands| {
        program(
            vec![data(tracked)],
            vec![(
                0,
                vec![encoder(Kind::Compute { concurrent }, &[], &[], commands)],
            )],
        )
    };
    let read_after_write = || vec![copy_word(0, 4), copy_word(4, 8)];
    assert!(problems(&compute(false, true, read_after_write())).is_empty());
    assert!(races(&compute(false, false, read_after_write())));
    // A concurrent one only across a barrier.
    assert!(races(&compute(true, true, read_after_write())));
    let barrier = vec![
        copy_word(0, 4),
        Command::Barrier { textures: false },
        copy_word(4, 8),
    ];
    assert!(problems(&compute(true, false, barrier)).is_empty());
}

#[test]
fn the_order_finds_races_between_encoders() {
    let two = |tracked, first: &[usize], second: &[usize]| {
        program(
            vec![data(tracked)],
            vec![(
                0,
                vec![
                    encoder(Kind::Blit, &[], first, vec![fill(0, 0, 8)]),
                    encoder(Kind::Blit, second, &[], vec![fill(0, 4, 12)]),
                ],
            )],
        )
    };
    // Tracking orders a tracked buffer; a fence orders an untracked one.
    assert!(problems(&two(true, &[], &[])).is_empty());
    assert!(races(&two(false, &[], &[])));
    assert!(problems(&two(false, &[0], &[0])).is_empty());
    // Tracking orders only one queue's encoders; an event orders two queues'.
    let queues = |tracked, event: bool| {
        let mut first = vec![encoder(Kind::Blit, &[], &[], vec![fill(0, 0, 8)])];
        let mut second = vec![encoder(Kind::Blit, &[], &[], vec![fill(0, 4, 12)])];
        if event {
            first.push(Step::Signal(0));
            second.insert(0, Step::Wait(0));
        }
        program(vec![data(tracked)], vec![(0, first), (1, second)])
    };
    assert!(races(&queues(true, false)));
    assert!(problems(&queues(false, true)).is_empty());
    // An event's wait before its signal, and a fence's wait with no update, are misuse.
    let early = program(
        vec![data(false)],
        vec![
            (0, vec![Step::Wait(0), Step::Signal(0)]),
            (0, vec![encoder(Kind::Blit, &[], &[], vec![fill(0, 0, 4)])]),
        ],
    );
    assert!(problems(&early)
        .iter()
        .any(|problem| problem.contains("event0")));
    let unfenced = program(
        vec![data(false)],
        vec![(0, vec![encoder(Kind::Blit, &[0], &[], vec![fill(0, 0, 4)])])],
    );
    assert!(problems(&unfenced)
        .iter()
        .any(|problem| problem.contains("fence0")));
}

#[test]
fn the_model_draws_whole_pixel_rectangles_and_mixes_as_the_fixture_does() {
    assert_eq!(mix(0, 0), 0x7F4A_7C15);
    assert_eq!(mix(1, 0), 0x9E37_79B1_u32.wrapping_add(0x7F4A_7C15));
    let mut rects = vec![0u8; BUFFER_BYTES as usize];
    for (index, word) in [3u32, 4, 1, 2].iter().enumerate() {
        rects[4 * index..4 * index + 4].copy_from_slice(&word.to_le_bytes());
    }
    let mut colors = vec![0u8; BUFFER_BYTES as usize];
    colors[..4].copy_from_slice(&0x1122_3344_u32.to_le_bytes());
    let program = Program {
        heaps: Vec::new(),
        buffers: vec![
            BufferSpec {
                role: Role::Rects,
                private: false,
                tracked: true,
                place: Place::Own,
                initial: rects,
            },
            BufferSpec {
                role: Role::Data,
                private: true,
                tracked: true,
                place: Place::Own,
                initial: colors,
            },
        ],
        textures: vec![TextureSpec {
            tracked: true,
            slices: 1,
            heap: false,
            initial: vec![0xEE; TEXTURE_BYTES as usize],
        }],
        views: Vec::new(),
        queues: 1,
        events: 0,
        fences: 0,
        command_buffers: vec![CommandBufferSpec {
            queue: 0,
            steps: vec![encoder(
                Kind::Render {
                    target: Surface::Texture(0),
                    slice: 0,
                    load: Load::Clear([1, 2, 3, 4]),
                },
                &[],
                &[],
                vec![Command::Draw {
                    rects: at(0, 0),
                    colors: at(1, 0),
                    instances: 1,
                    indirect: None,
                }],
            )],
        }],
    };
    assert!(problems(&program).is_empty());
    let state = run(&program);
    for y in 0..8u64 {
        for x in 0..8u64 {
            let start = (y * ROW_BYTES + x * TEXEL_BYTES) as usize;
            let inside = (1..3).contains(&x) && (2..4).contains(&y);
            let want: [u8; 4] = if inside {
                [0x44, 0x33, 0x22, 0x11]
            } else {
                [1, 2, 3, 4]
            };
            assert_eq!(
                state.textures[0][start..start + 4],
                want,
                "texel ({x}, {y})"
            );
        }
    }
}

/// A heap of `tracked` mode whose buffers 0 and 1 lie at offsets 0 and 256: their bytes 256..512
/// and 0..256 are one memory.
fn aliased(tracked: bool, command_buffers: Vec<(usize, Vec<Step>)>) -> Program {
    let placed = |offset| BufferSpec {
        role: Role::Data,
        private: false,
        tracked,
        place: Place::Heap { heap: 0, offset },
        initial: Vec::new(),
    };
    let mut program = program(vec![placed(0), placed(256)], command_buffers);
    program.heaps.push(HeapSpec {
        private: false,
        tracked,
        initial: (0..HEAP_WINDOW).map(|byte| byte as u8).collect(),
    });
    program
}

#[test]
fn the_order_takes_memory_and_objects() {
    // Two heap buffers over the same bytes conflict: in one blit encoder they race; disjoint
    // bytes of them do not.
    let fills = |second: (u64, u64)| {
        aliased(
            false,
            vec![(
                0,
                vec![encoder(
                    Kind::Blit,
                    &[],
                    &[],
                    vec![fill(0, 256, 260), fill(1, second.0, second.1)],
                )],
            )],
        )
    };
    assert!(races(&fills((0, 4))));
    assert!(problems(&fills((4, 8))).is_empty());
    // Across encoders of one queue, a tracked heap orders them as one resource; an untracked
    // one does not.
    let encoders = |tracked| {
        aliased(
            tracked,
            vec![(
                0,
                vec![
                    encoder(Kind::Blit, &[], &[], vec![fill(0, 256, 260)]),
                    encoder(Kind::Blit, &[], &[], vec![fill(1, 0, 4)]),
                ],
            )],
        )
    };
    assert!(problems(&encoders(true)).is_empty());
    assert!(races(&encoders(false)));
    // Metal's API validation raises on a copy between two buffers over bytes in common, even of
    // bytes apart: buffer 0's bytes 0..4 and buffer 1's 256..260 are the heap's 0..4 and 512..516.
    let between = |to: usize| {
        aliased(
            false,
            vec![(
                0,
                vec![encoder(
                    Kind::Blit,
                    &[],
                    &[],
                    vec![Command::Copy {
                        from: at(0, 0),
                        to: at(to, 256),
                        size: 4,
                    }],
                )],
            )],
        )
    };
    let refused = problems(&between(1));
    assert!(
        refused.len() == 1 && refused[0].contains("over bytes in common"),
        "{refused:?}"
    );
    assert!(problems(&between(0)).is_empty());
    // A view and its texture are objects of their own: tracking is not relied on between them,
    // so a write through the view and a read through the texture need a fence.
    let view_then_texture = |fence: bool| {
        let (updates, waits): (&[usize], &[usize]) = if fence { (&[0], &[0]) } else { (&[], &[]) };
        let mut program = program(
            vec![data(true)],
            vec![(
                0,
                vec![
                    encoder(
                        Kind::Blit,
                        &[],
                        updates,
                        vec![Command::ToTexture {
                            from: at(0, 0),
                            to: Surface::View(0),
                            slice: 0,
                            rows: (0, 2),
                        }],
                    ),
                    encoder(
                        Kind::Blit,
                        waits,
                        &[],
                        vec![Command::FromTexture {
                            from: Surface::Texture(0),
                            slice: 1,
                            rows: (1, 3),
                            to: at(0, 256),
                        }],
                    ),
                ],
            )],
        );
        program.textures.push(TextureSpec {
            tracked: true,
            slices: 2,
            heap: false,
            initial: vec![0; 2 * TEXTURE_BYTES as usize],
        });
        program.views.push(ViewSpec {
            texture: 0,
            slice: 1,
            format: Format::Bgra8Unorm,
            form: ViewForm::Slices,
        });
        program
    };
    assert!(races(&view_then_texture(false)));
    assert!(problems(&view_then_texture(true)).is_empty());
}

#[test]
fn the_model_reads_heaps_and_views_as_metal_does() {
    // A word copied through heap buffer 1 lands at its offset in the heap, under buffer 0's
    // bytes 256 on.
    let copied = aliased(
        true,
        vec![(
            0,
            vec![encoder(
                Kind::Compute { concurrent: false },
                &[],
                &[],
                vec![Command::Dispatch {
                    kernel: Kernel::Copy,
                    a: at(0, 0),
                    b: at(1, 8),
                    indirect: None,
                }],
            )],
        )],
    );
    assert!(problems(&copied).is_empty());
    let state = run(&copied);
    assert_eq!(state.heaps[0][264..268], [0, 1, 2, 3]);
    // A clear and a texel write through a BGRA view store red and blue swapped; a swizzled
    // view's reads take its channels.
    let mut colors = vec![0u8; BUFFER_BYTES as usize];
    colors[..4].copy_from_slice(&[0x10, 0x20, 0x30, 0x40]);
    let mut viewed = program(
        vec![BufferSpec {
            initial: colors,
            ..data(true)
        }],
        vec![(
            0,
            vec![
                encoder(
                    Kind::Render {
                        target: Surface::View(0),
                        slice: 0,
                        load: Load::Clear([1, 2, 3, 4]),
                    },
                    &[],
                    &[0],
                    vec![],
                ),
                encoder(
                    Kind::Compute { concurrent: true },
                    &[0],
                    &[],
                    vec![
                        Command::TexelWrite {
                            from: at(0, 0),
                            to: Surface::View(0),
                            rows: 1,
                        },
                        Command::Barrier { textures: true },
                        Command::TexelRead {
                            from: Surface::View(1),
                            rows: 1,
                            to: at(0, 256),
                        },
                    ],
                ),
            ],
        )],
    );
    viewed.textures.push(TextureSpec {
        tracked: false,
        slices: 1,
        heap: true,
        initial: vec![0; TEXTURE_BYTES as usize],
    });
    viewed.views.push(ViewSpec {
        texture: 0,
        slice: 0,
        format: Format::Bgra8Unorm,
        form: ViewForm::Whole,
    });
    viewed.views.push(ViewSpec {
        texture: 0,
        slice: 0,
        format: Format::Rgba8Unorm,
        form: ViewForm::Swizzled([Channel::Blue, Channel::One, Channel::Zero, Channel::Red]),
    });
    assert!(problems(&viewed).is_empty());
    let state = run(&viewed);
    // Texel (0, 0): written from the word (red 0x10, blue 0x30), stored blue first.
    assert_eq!(state.textures[0][..4], [0x30, 0x20, 0x10, 0x40]);
    // Texel (1, 0): the word there is zero.
    assert_eq!(state.textures[0][4..8], [0, 0, 0, 0]);
    // Row 1: the clear, (1, 2, 3, 4) stored blue first.
    assert_eq!(
        state.textures[0][ROW_BYTES as usize..ROW_BYTES as usize + 4],
        [3, 2, 1, 4]
    );
    // The swizzled read of texel (0, 0) as stored (0x30, 0x20, 0x10, 0x40): blue, one, zero,
    // red.
    assert_eq!(state.buffers[0][256..260], [0x10, 0xFF, 0, 0x30]);
}

#[test]
fn tidy_drops_what_a_removal_left_without_purpose() {
    // The fence's updating encoder is gone: its wait goes, and the event left numbered from 0.
    let program = program(
        vec![data(true)],
        vec![
            (
                0,
                vec![
                    encoder(Kind::Blit, &[], &[1], vec![]),
                    encoder(Kind::Blit, &[1], &[], vec![fill(0, 0, 4)]),
                    Step::Signal(3),
                ],
            ),
            (
                0,
                vec![
                    Step::Wait(3),
                    encoder(Kind::Blit, &[], &[], vec![fill(0, 4, 8)]),
                ],
            ),
            (0, vec![Step::Wait(2)]),
        ],
    );
    let tidied = tidy(program);
    assert_eq!(tidied.fences, 0);
    assert_eq!(tidied.events, 1);
    assert_eq!(tidied.command_buffers.len(), 2);
    assert_eq!(
        tidied.command_buffers[0].steps,
        vec![
            encoder(Kind::Blit, &[], &[], vec![fill(0, 0, 4)]),
            Step::Signal(0)
        ]
    );
    assert_eq!(tidied.command_buffers[1].steps[0], Step::Wait(0));
}

/// The minimizer keeps a failure (here: a command the sequence must still hold), a race-free
/// sequence, and stops only where no single removal keeps both.
#[test]
fn the_minimizer_keeps_the_failure_in_a_race_free_sequence() {
    let mut tried = 0;
    for seed in 0..SEEDS {
        let program = generate(seed);
        if program.command_count() < 6 {
            continue;
        }
        let kept = program
            .encoders()
            .flat_map(|(_, _, encoder)| encoder.commands.iter())
            .filter(|command| !matches!(command, Command::Barrier { .. }))
            .last()
            .expect("a command")
            .clone();
        let holds = |candidate: &Program| {
            candidate
                .encoders()
                .any(|(_, _, encoder)| encoder.commands.contains(&kept))
        };
        let minimized = minimize(&program, 10_000, &holds);
        assert!(minimized.attempts < 10_000, "seed {seed}");
        let smaller = &minimized.program;
        assert!(problems(smaller).is_empty(), "seed {seed}:\n{smaller}");
        assert!(holds(smaller), "seed {seed}:\n{smaller}");
        assert!(
            smaller.command_count() < program.command_count(),
            "seed {seed}:\n{smaller}"
        );
        for candidate in removals(smaller) {
            let candidate = tidy(candidate);
            let removed = candidate.command_buffers != smaller.command_buffers;
            assert!(
                !removed || !problems(&candidate).is_empty() || !holds(&candidate),
                "seed {seed}: a removal from the minimized sequence keeps the failure:\n{candidate}"
            );
        }
        tried += 1;
        if tried == 8 {
            return;
        }
    }
    panic!("fewer than 8 seeds of 6 commands or more");
}

/// Every committed line is current, records every seed's every resource or the seed's Metal-side
/// gap alone ([`METAL_GAP`]: Metal's driver crashed running it), names each record once and of a
/// seed, and its bytes are the model's: Metal checks the model, so a runtime difference is the
/// runtime's.
#[test]
fn committed_records_are_current_and_agree_with_the_model() {
    let lines = read_lines(&records_path(NAME)).unwrap();
    let mut problems = Vec::new();
    if lines.is_empty() {
        problems.push(format!("{NAME}: no Metal record"));
    }
    let seeds: Vec<(String, Program)> = (0..SEEDS)
        .map(|seed| (format!("{}/", case_name(seed)), generate(seed)))
        .collect();
    for line in &lines {
        let environment = &line.fingerprint.metal_environment;
        let stale = stale(&line.fingerprint, &fingerprint(environment).unwrap());
        if !stale.is_empty() {
            problems.push(format!(
                "{NAME} on {environment}: stale ({}): record it again",
                stale.join(", ")
            ));
            continue;
        }
        let mut names = std::collections::BTreeSet::new();
        for (name, _) in &line.records {
            if !names.insert(name) {
                problems.push(format!("{NAME} on {environment}: {name} is recorded twice"));
            }
            if !seeds.iter().any(|(prefix, _)| name.starts_with(prefix)) {
                problems.push(format!("{NAME} on {environment}: {name} is of no seed"));
            }
        }
        for (seed, (prefix, program)) in seeds.iter().enumerate() {
            if let Some(detail) = metal_gap(&line.records, seed as u64) {
                eprintln!("{NAME} on {environment}: seed {seed} is a Metal-side gap: {detail}");
                continue;
            }
            let expected = model_records(&prefix[..prefix.len() - 1], program);
            for difference in byte_differences(&line.records, &expected) {
                problems.push(format!(
                    "{NAME} on {environment}: seed {seed}: {difference}"
                ));
            }
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

/// A seed's Metal-side gap is its one record, a gap refusal with a detail; anything else beside
/// it, or another kind of refusal, is no gap.
#[test]
fn a_metal_gap_is_a_seeds_lone_gap_record() {
    let gap = |detail: &str| Value::Refused {
        refusal: Refusal::Gap,
        detail: detail.to_owned(),
    };
    let name = |seed: u64, call: &str| format!("{}/{call}", case_name(seed));
    let lone = vec![
        (name(3, METAL_GAP), gap("crashed")),
        (name(4, "cb0"), Value::Done),
    ];
    assert_eq!(metal_gap(&lone, 3), Some("crashed"));
    assert_eq!(metal_gap(&lone, 4), None);
    assert_eq!(metal_gap(&lone, 5), None);
    let beside = vec![
        (name(3, METAL_GAP), gap("crashed")),
        (name(3, "cb0"), Value::Done),
    ];
    assert_eq!(metal_gap(&beside, 3), None);
    assert_eq!(metal_gap(&[(name(3, METAL_GAP), gap(""))], 3), None);
    assert_eq!(metal_gap(&[(name(3, "cb0"), gap("crashed"))], 3), None);
    let failed = Value::Refused {
        refusal: Refusal::Failed,
        detail: "crashed".to_owned(),
    };
    assert_eq!(metal_gap(&[(name(3, METAL_GAP), failed)], 3), None);
}

/// The layer's hazards, as it words them at a submission and at recording, and what else it says.
#[test]
fn the_lane_reads_the_layers_hazards() {
    let submitted = "vkQueueSubmit(): READ_AFTER_WRITE hazard detected. vkCmdDrawIndirect (from \
        VkCommandBuffer 0x7f63940015a0 submitted on the current VkQueue 0x7f63f4e36bf0) reads \
        VkBuffer 0x500000000050, which was previously written by vkCmdDispatch (from \
        VkCommandBuffer 0x7f63f49f48f0 submitted on VkQueue 0x7f63f4e36bf0). \nThe current \
        synchronization allows 0, but to prevent this hazard, it must allow \
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT accesses at VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT.";
    assert_eq!(
        parse("SYNC-HAZARD-READ-AFTER-WRITE", submitted),
        Some(Hazard {
            later_writes: false,
            earlier_writes: true,
            later: "vkCmdDrawIndirect".to_owned(),
            earlier: "vkCmdDispatch".to_owned(),
            memory: Named::Buffer(0x5000_0000_0050),
            one_cb: false,
        })
    );
    let another = "vkQueueSubmit(): WRITE_AFTER_READ hazard detected. vkCmdDispatch (from \
        VkCommandBuffer 0x1 submitted on the current VkQueue 0x2) writes to VkBuffer 0x5d, which \
        was previously read by another vkCmdDispatch command (from VkCommandBuffer 0x3 submitted \
        on VkQueue 0x2).";
    let read = parse("SYNC-HAZARD-WRITE-AFTER-READ", another).unwrap();
    assert!(read.later_writes && !read.earlier_writes);
    assert_eq!(read.earlier, "vkCmdDispatch");
    assert_eq!(read.memory, Named::Buffer(0x5d));
    let recorded = "vkCmdCopyBufferToImage(): WRITE_AFTER_WRITE hazard detected. \
        vkCmdCopyBufferToImage writes to VkImage 0x560000000056, which was previously written by \
        vkCmdDispatch.";
    let image = parse("SYNC-HAZARD-WRITE-AFTER-WRITE", recorded).unwrap();
    assert_eq!(image.later, "vkCmdCopyBufferToImage");
    assert_eq!(image.memory, Named::Image(0x5600_0000_0056));
    // A transfer command's access at recording names its parameter before the memory (lavapipe,
    // `memoryless_uses`).
    let filled = "vkCmdFillBuffer(): WRITE_AFTER_WRITE hazard detected. vkCmdFillBuffer writes \
        to dstBuffer VkBuffer 0x700000000070, which was previously written by vkCmdDispatch. \nNo \
        sufficient synchronization is present to ensure that a write \
        (VK_ACCESS_2_TRANSFER_WRITE_BIT) at VK_PIPELINE_STAGE_2_CLEAR_BIT does not conflict with \
        a prior write (VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT) at \
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT.\nBuffer access region: {\n  offset = 12\n  size \
        = 4\n}\n";
    assert_eq!(
        parse("SYNC-HAZARD-WRITE-AFTER-WRITE", filled),
        Some(Hazard {
            later_writes: true,
            earlier_writes: true,
            later: "vkCmdFillBuffer".to_owned(),
            earlier: "vkCmdDispatch".to_owned(),
            memory: Named::Buffer(0x7000_0000_0070),
            one_cb: true,
        })
    );
    let source = recorded.replace("writes to VkImage", "reads srcImage VkImage");
    let copied = parse("SYNC-HAZARD-READ-AFTER-WRITE", &source).unwrap();
    assert_eq!(copied.memory, Named::Image(0x5600_0000_0056));
    // Layer 1.4.363 names a shader's access through a descriptor by its stage: before its command
    // at a submission (seen on MoltenVK 1.4.2), alone at recording, where the call is the command.
    let staged_recorded = "vkCmdDispatch(): WRITE_AFTER_WRITE hazard detected. Shader stage \
        VK_SHADER_STAGE_COMPUTE_BIT writes to VkBuffer 0x4c000000004c, which was previously \
        written by vkCmdCopyBuffer.";
    assert_eq!(
        parse("SYNC-HAZARD-WRITE-AFTER-WRITE", staged_recorded),
        Some(Hazard {
            later_writes: true,
            earlier_writes: true,
            later: "vkCmdDispatch".to_owned(),
            earlier: "vkCmdCopyBuffer".to_owned(),
            memory: Named::Buffer(0x4c00_0000_004c),
            one_cb: true,
        })
    );
    let staged = "vkQueueSubmit(): pSubmits[0].pCommandBuffers[4] READ_AFTER_WRITE hazard \
        detected. Shader stage VK_SHADER_STAGE_COMPUTE_BIT in vkCmdDispatch (from VkCommandBuffer \
        0x7733041018 submitted on the current VkQueue 0x7732cfef98) reads VkBuffer 0x4c000000004c, \
        which was previously written by another vkCmdDispatch command (from VkCommandBuffer \
        0x7733040618 submitted on VkQueue 0x7732cfef98).";
    assert_eq!(
        parse("SYNC-HAZARD-READ-AFTER-WRITE", staged),
        Some(Hazard {
            later_writes: false,
            earlier_writes: true,
            later: "vkCmdDispatch".to_owned(),
            earlier: "vkCmdDispatch".to_owned(),
            memory: Named::Buffer(0x4c00_0000_004c),
            one_cb: false,
        })
    );
    // At a recording, or with no barrier at all after an earlier write, the two commands are in
    // one command buffer; a read with none may be in an earlier batch.
    assert!(image.one_cb && !read.one_cb);
    let unbarriered = |id_name: &str, text: &str| {
        let text = format!(
            "{text}\nNo sufficient synchronization is present to ensure that a write \
             (VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT) at VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT \
             does not conflict with a prior write of the same type at the same stage."
        );
        parse(id_name, &text).unwrap().one_cb
    };
    let written = another
        .replace("WRITE_AFTER_READ", "WRITE_AFTER_WRITE")
        .replace("previously read by", "previously written by");
    assert!(unbarriered("SYNC-HAZARD-WRITE-AFTER-WRITE", &written));
    assert!(!unbarriered("SYNC-HAZARD-WRITE-AFTER-READ", another));
    // Another message, a hazard of another wording, or a direction the id does not name.
    assert_eq!(parse("VUID-vkCmdDraw-None-08600", submitted), None);
    assert_eq!(
        parse("SYNC-HAZARD-READ-AFTER-WRITE", "a hazard of no shape"),
        None
    );
    assert_eq!(parse("SYNC-HAZARD-WRITE-AFTER-READ", submitted), None);
}

/// The handles a run bound, matched with the sequence's commands in encoded order: a heap's
/// buffers are its memory, and a run that encoded otherwise names nothing.
#[test]
fn the_lane_names_the_memory_a_runs_buffers_bound() {
    let mut heaped = data(false);
    heaped.place = Place::Heap {
        heap: 0,
        offset: 256,
    };
    let mut sequence = program(
        vec![data(true), heaped],
        vec![(
            0,
            vec![
                encoder(Kind::Blit, &[], &[], vec![fill(1, 0, 8)]),
                encoder(
                    Kind::Compute { concurrent: true },
                    &[],
                    &[],
                    vec![copy_word(0, 4), Command::Barrier { textures: false }],
                ),
            ],
        )],
    );
    sequence.heaps.push(HeapSpec {
        private: false,
        tracked: false,
        initial: vec![0; HEAP_WINDOW as usize],
    });
    let call = |committed, call, buffers: &[u64]| Encoded {
        committed,
        call,
        buffers: buffers.to_vec(),
    };
    let setup = call(0, Call::Copy, &[9, 10]);
    let encoded = [
        setup.clone(),
        call(1, Call::Fill, &[0x20]),
        call(1, Call::Dispatch, &[0x10, 0x10]),
        call(2, Call::Copy, &[0x10, 11]),
    ];
    let handles = Handles::of(&sequence, &encoded);
    let named_by = |handles: &Result<Handles, String>, raw: u64| {
        verdict(
            &sequence,
            &Order::new(&sequence),
            &accesses(&sequence),
            handles,
            &Hazard {
                later_writes: true,
                earlier_writes: true,
                later: "vkCmdDispatch".to_owned(),
                earlier: "vkCmdFillBuffer".to_owned(),
                memory: Named::Buffer(raw),
                one_cb: false,
            },
            &BTreeSet::from([0]),
        )
    };
    let named = |raw: u64| named_by(&handles, raw).to_string();
    // Each memory has one writing command, so no pair: named, but not explained.
    assert!(named(0x20).contains("no two commands"), "{}", named(0x20));
    assert!(named(0x10).contains("no two commands"), "{}", named(0x10));
    // The setup's and the readback's buffers are none of the sequence's.
    assert!(named(9).contains("is no buffer"), "{}", named(9));
    assert!(named(11).contains("is no buffer"), "{}", named(11));
    // A run a refusal ended: the calls it made.
    let prefix = Handles::of(&sequence, &encoded[..2]);
    assert!(named_by(&prefix, 0x20)
        .to_string()
        .contains("no two commands"));
    assert!(named_by(&prefix, 0x10).to_string().contains("is no buffer"));
    // A run that encoded other calls, or more, names nothing.
    let wrong = Handles::of(&sequence, &[setup, call(1, Call::Copy, &[0x20])]);
    assert!(wrong.is_err());
    assert!(named_by(&wrong, 0x20)
        .to_string()
        .contains("the run encoded Copy"));
    let past = [
        call(1, Call::Fill, &[0x20]),
        call(1, Call::Dispatch, &[0x10, 0x10]),
        call(1, Call::Draw, &[0x10, 0x10]),
    ];
    assert!(Handles::of(&sequence, &past).is_err());
}

/// A hazard between two commands that share a byte is the runtime's to explain; one between
/// commands whose bytes are apart is explained, with what Metal orders between them.
#[test]
fn a_hazard_is_explained_only_when_its_commands_share_no_byte() {
    let two = |tracked: bool, concurrent: bool, apart: bool, encoders: bool| {
        let read = if apart { 300 } else { 256 };
        let kind = Kind::Compute { concurrent };
        let steps = if encoders {
            vec![
                encoder(kind, &[], &[], vec![copy_word(0, 256)]),
                encoder(kind, &[], &[], vec![copy_word(read, 400)]),
            ]
        } else {
            vec![encoder(
                kind,
                &[],
                &[],
                vec![copy_word(0, 256), copy_word(read, 400)],
            )]
        };
        program(vec![data(tracked)], vec![(0, steps)])
    };
    let raw = 0x50;
    let hazard = Hazard {
        later_writes: false,
        earlier_writes: true,
        later: "vkCmdDispatch".to_owned(),
        earlier: "vkCmdDispatch".to_owned(),
        memory: Named::Buffer(raw),
        one_cb: false,
    };
    let judge_in = |sequence: &Program, hazard: &Hazard, later_cbs: &[usize]| {
        let encoded: Vec<Encoded> = sequence
            .encoders()
            .flat_map(|(cb, _, encoder)| encoder.commands.iter().map(move |_| cb))
            .map(|cb| Encoded {
                committed: cb + 1,
                call: Call::Dispatch,
                buffers: vec![raw, raw],
            })
            .collect();
        verdict(
            sequence,
            &Order::new(sequence),
            &accesses(sequence),
            &Handles::of(sequence, &encoded),
            hazard,
            &later_cbs.iter().copied().collect(),
        )
    };
    let judge = |sequence: &Program, hazard: &Hazard| judge_in(sequence, hazard, &[0]);
    let metal = |verdict: Verdict| match verdict {
        Verdict::Explained { metal, .. } => metal.into_iter().collect::<Vec<_>>(),
        other => panic!("not explained: {other}"),
    };
    // Untracked memory in two encoders: Metal leaves the order to the program.
    assert!(matches!(
        judge(&two(false, false, false, true), &hazard),
        Verdict::Shared { .. }
    ));
    assert_eq!(
        metal(judge(&two(false, false, true, true), &hazard)),
        [MetalOrder::Untracked]
    );
    // A tracked buffer in two encoders, or in one serial encoder: Metal orders them.
    assert_eq!(
        metal(judge(&two(true, false, true, true), &hazard)),
        [MetalOrder::Tracked]
    );
    assert_eq!(
        metal(judge(&two(true, false, true, false), &hazard)),
        [MetalOrder::Ordered]
    );
    // One concurrent encoder: the program's, through a memory barrier.
    assert_eq!(
        metal(judge(&two(true, true, true, false), &hazard)),
        [MetalOrder::Concurrent]
    );
    let shared = judge(&two(true, true, false, false), &hazard);
    assert!(
        shared
            .to_string()
            .contains("cb0 s0 c0 writes buffer0 bytes 256..260")
            && shared
                .to_string()
                .contains("cb0 s0 c1 reads buffer0 bytes 256..260"),
        "{shared}"
    );
    // A hazard no two commands of the sequence can make is not explained.
    let lone = program(
        vec![data(false)],
        vec![(
            0,
            vec![encoder(
                Kind::Compute { concurrent: false },
                &[],
                &[],
                vec![copy_word(0, 256)],
            )],
        )],
    );
    assert!(matches!(judge(&lone, &hazard), Verdict::Unnamed(_)));
    // Nor is one whose commands the runtime records with other Vulkan commands than it names, or
    // whose later command is in a command buffer the layer did not check then.
    let copied = Hazard {
        later: "vkCmdCopyBuffer".to_owned(),
        ..hazard.clone()
    };
    let shared = two(false, false, false, true);
    assert!(matches!(judge(&shared, &copied), Verdict::Unnamed(_)));
    let unnamed = judge_in(&shared, &hazard, &[1]).to_string();
    assert!(unnamed.contains("the later in cb1"), "{unnamed}");
    // Nor one in a command buffer of its own whose commands are in two.
    let step = |command| encoder(Kind::Compute { concurrent: false }, &[], &[], vec![command]);
    let two_cbs = program(
        vec![data(false)],
        vec![
            (0, vec![step(copy_word(0, 256))]),
            (0, vec![step(copy_word(256, 400))]),
        ],
    );
    assert!(matches!(
        judge_in(&two_cbs, &hazard, &[1]),
        Verdict::Shared { .. }
    ));
    let one_cb = Hazard {
        one_cb: true,
        ..hazard.clone()
    };
    assert!(matches!(
        judge_in(&two_cbs, &one_cb, &[1]),
        Verdict::Unnamed(_)
    ));
}

/// The Vulkan commands the runtime records each command with, as the layer names them.
#[test]
fn the_lane_names_each_commands_vulkan_commands() {
    let names = |command: &Command| vulkan_commands(Some(command));
    assert!(names(&fill(0, 0, 8)).contains(&"vkCmdFillBuffer"));
    assert!(names(&fill(0, 0, 8)).contains(&"vkCmdUpdateBuffer"));
    assert!(names(&copy_word(0, 4)).contains(&"vkCmdDispatchIndirect"));
    assert!(!names(&copy_word(0, 4)).contains(&"vkCmdDraw"));
    assert!(names(&Command::Barrier { textures: false }).is_empty());
    // A render pass's own accesses: its beginning's, its end's, its draws'; no dispatch's.
    let pass = vulkan_commands(None);
    assert!(pass.contains(&"vkCmdEndRenderPass") && pass.contains(&"vkCmdDraw"));
    assert!(!pass.contains(&"vkCmdDispatch"));
}

/// An image the layer names is any texture: its two commands share a byte if any texture's do.
#[test]
fn an_images_hazard_is_read_on_every_texture() {
    let texture = |tracked| TextureSpec {
        tracked,
        slices: 1,
        heap: false,
        initial: vec![0; TEXTURE_BYTES as usize],
    };
    let rows = |to: Surface, rows: (u32, u32)| Command::ToTexture {
        from: at(0, 0),
        to,
        slice: 0,
        rows,
    };
    let write = |to| Command::TexelWrite {
        from: at(0, 0),
        to,
        rows: 2,
    };
    let sequence = |late: (u32, u32)| {
        let mut sequence = program(
            vec![data(false)],
            vec![(
                0,
                vec![
                    encoder(
                        Kind::Compute { concurrent: false },
                        &[],
                        &[],
                        vec![write(Surface::Texture(0)), write(Surface::Texture(1))],
                    ),
                    encoder(Kind::Blit, &[], &[], vec![rows(Surface::Texture(1), late)]),
                ],
            )],
        );
        sequence.textures = vec![texture(true), texture(false)];
        sequence
    };
    let hazard = Hazard {
        later_writes: true,
        earlier_writes: true,
        later: "vkCmdCopyBufferToImage".to_owned(),
        earlier: "vkCmdDispatch".to_owned(),
        memory: Named::Image(0x56),
        one_cb: false,
    };
    let judge = |sequence: &Program| {
        verdict(
            sequence,
            &Order::new(sequence),
            &accesses(sequence),
            &Err("unused".to_owned()),
            &hazard,
            &BTreeSet::from([0]),
        )
    };
    let apart = judge(&sequence((5, 6)));
    assert!(
        matches!(&apart, Verdict::Explained { metal, .. } if metal.contains(&MetalOrder::Untracked)),
        "{apart}"
    );
    assert!(matches!(judge(&sequence((1, 3))), Verdict::Shared { .. }));
}
