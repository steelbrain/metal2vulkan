//! What Metal writes into a pass's visibility-result buffer: the words a `Boolean` mode stores,
//! the samples a `Counting` mode counts, and what an offset nobody set keeps.
//!
//! Ported from the runtime lane `runtime_lane/visibility.rs`. Every pass here is one the lane
//! drives, drawn with the owned fixture `rt6_render_state`: the fullscreen triangle covers every
//! pixel of a 4 by 4 target, and the same draw **from vertex 3** covers none. Each visibility
//! buffer starts as `0xaa` bytes, so a word Metal never wrote is visible as the bytes that were
//! there rather than as a zero nobody can tell from a result; an `Accumulate` shape's starts as
//! the words it adds to.
//!
//! The lane's own table was measured on an M4, and its comment says so. Metal's answer here is
//! **run**, not copied: record mode takes it on this M3 Max, and this module's tests cross-check
//! the recordings against the lane's words, which is the evidence the port owes the lane. A fresh
//! answer that disagrees with the lane is a finding about one of the two chips, not a nuisance.
//!
//! Every word compares exactly but a `Boolean` result's: the M3 Max writes 1 where a sample
//! passed, and the runtime the device's own nonzero value, which it announces
//! (`device-boolean-visibility`), whose documentation states the bound the comparison holds: zero
//! exactly where Metal's word is zero ([`Loosening::Nonzero`] at that word alone, `boolean_words`).
//! Not a native note of the suite's set: the note's bound is exact, so a case names it at the
//! words it loosens.
//!
//! The shapes, each its own case so a device that lacks precise occlusion queries loses only the
//! counted rows:
//!
//! - `Boolean` at a spread of offsets: drawn, a draw covering nothing, an offset set with no draw
//!   after it, `Disabled` at another offset, and two draws at one offset; then a second pass over
//!   the same buffer, which rewrites only its own offset.
//! - An offset taken again after another offset, in both orders.
//! - A draw behind the cleared depth, which passes no sample: the one shape here with a **depth
//!   attachment** and a depth/stencil state.
//! - An indirect draw of no instances.
//! - The same questions counted, and the fullscreen triangle's count at **4x**, which is the one
//!   shape here with a multisampled attachment.
//! - `Accumulate` (`visibilityResultType`), counted: a draw's samples added to the word, a word
//!   set with no draw kept, a sum past 32 and past 64 bits, a second pass in the next command
//!   buffer and in the **same** one (`joined`, read once after both), and an offset taken again.
//!   `Boolean`: words 0, 2 and 3 under a covering draw and under one covering nothing.
//!
//! # How this family's inputs are keyed, and the one key the compiler cannot guard
//!
//! Every input fingerprint here is **the name of the Metal call and the values handed it**, never
//! `Debug` text. Three of them were `format!("{:?}")`, and one was the worst instance of that trap
//! in the suite: the depth/stencil state keyed on
//! [`metal2vulkan::metal::state::DepthStencilDescriptor`], a public type of the **published**
//! translator crate, so a field added to that struct in another crate staled these recordings and
//! nothing in either crate connected the two.
//!
//! `Act` and `Depth` are this module's own types, so they are matched and destructured with
//! **no `..`**: a field added to either is a compile error here, which is where the decision
//! belongs.
//!
//! **The depth/stencil key cannot have that protection, and the substitute is not a fallback.**
//! `DepthStencilDescriptor` and `StencilDescriptor` are `#[non_exhaustive]`, so this crate cannot
//! destructure them field by field — the compiler *requires* a `..` — and there is no compile
//! error to be had across that boundary. So `depth_state_words` states the one field this family
//! sets and **refuses rather than keys** a descriptor that differs from
//! `DepthStencilDescriptor::default()` anywhere else. That gives both halves of what is wanted,
//! and it holds without any help from the compiler: a field this family starts setting surfaces as
//! an error out of `inputs()` instead of a key quietly missing it, while a field left at its
//! default writes nothing — which is the property that actually stops a widened descriptor from
//! staling these recordings.
//!
//! //! Nothing here can fault a GPU: one 4 by 4 target a pass, three vertices a draw, no loop.

use crate::allowance::{Allowance, Justification, Loosening};
use crate::case::{Abort, Case, Failure, Handle, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::render::{
    ColorState, PassColor, PassDepth, RenderPass, RenderPipelineSpec, RenderStep, RenderWork,
    TargetSpec,
};
use crate::replay::NOTICES_SITE;
use crate::transcript::{ByteLayout, Digest, Element, Node, Observed, Sites, Value};
use metal2vulkan::metal::state::DepthStencilDescriptor;
use metal2vulkan::metal::{CompareFunction, PixelFormat, PrimitiveType};
use metal2vulkan_runtime::{Degradation, StoreAction, VisibilityResultMode, VisibilityResultType};
use std::collections::BTreeSet;

/// The owned fixture the lane draws with: `rt6_full` takes a depth and a slope as one `float2` at
/// vertex buffer 0, and `rt6_white` returns white.
const FIXTURE: &str = "rt6_render_state";
const VERTEX: &str = "rt6_full";
const FRAGMENT: &str = "rt6_white";

/// The lane's target: a 4 by 4 `RGBA8Unorm` render target, whose 16 pixels are the 16 samples a
/// covering draw passes at 1x.
const SIDE: u32 = 4;
const COLOR: PixelFormat = PixelFormat::RGBA8Unorm;

/// The byte every visibility buffer starts as, so a word Metal never wrote is readable as itself.
const UNWRITTEN: u8 = 0xaa;

/// The depth the lane's draws take unless the shape says otherwise.
const DEPTH: f32 = 0.5;

/// One command a case encodes into a pass, in the lane's own vocabulary.
#[derive(Clone, Copy, Debug)]
enum Act {
    /// `setVisibilityResultMode:offset:`.
    Mode(VisibilityResultMode, u64),
    /// The fullscreen triangle at `z`, or the same draw from vertex 3, which covers no pixel, of
    /// `instances` instances.
    Draw {
        z: f32,
        covering: bool,
        instances: u32,
    },
    /// `drawPrimitives:indirectBuffer:indirectBufferOffset:` of three vertices and **no**
    /// instances, which the lane's arguments buffer holds.
    DrawIndirect,
}

impl Act {
    /// The fullscreen triangle at the lane's depth, covering every pixel.
    const fn draw() -> Self {
        Act::Draw {
            z: DEPTH,
            covering: true,
            instances: 1,
        }
    }

    /// The same draw from vertex 3, which covers no pixel.
    const fn draw_nothing() -> Self {
        Act::Draw {
            z: DEPTH,
            covering: false,
            instances: 1,
        }
    }
}

/// The depth state one shape takes: a compare function, with no depth write, over a depth
/// attachment of `format` cleared to `clear`.
#[derive(Clone, Copy, Debug)]
struct Depth {
    format: PixelFormat,
    clear: f64,
    compare: CompareFunction,
}

/// What one case asks: its name, its buffer's words, the passes it runs, its depth attachment, its
/// sample count, and its passes' visibility result type, starting words and command buffers.
struct Shape {
    name: &'static str,
    /// 64-bit words the case's visibility buffer holds.
    words: usize,
    /// The passes, in order, each over its own fresh color target and all over **one** visibility
    /// buffer, as the lane runs them.
    passes: &'static [&'static [Act]],
    depth: Option<Depth>,
    /// `rasterSampleCount` and the attachment's samples: 1, or 4 for the lane's 4x pass.
    samples: u32,
    /// Every pass's `visibilityResultType`.
    kind: VisibilityResultType,
    /// The words the buffer starts as, `words` of them, or `None` for [`UNWRITTEN`] bytes.
    start: Option<&'static [u64]>,
    /// Whether the passes run in **one** command buffer, whose words are read once after it,
    /// rather than one command buffer a pass, read after each.
    joined: bool,
}

/// `Boolean`, `Counting` and `Disabled`, as short as the lane writes them.
const BOOL: VisibilityResultMode = VisibilityResultMode::Boolean;
const COUNT: VisibilityResultMode = VisibilityResultMode::Counting;
const OFF: VisibilityResultMode = VisibilityResultMode::Disabled;

/// `Reset`, Metal's default, and `Accumulate`.
const RESET: VisibilityResultType = VisibilityResultType::Reset;
const ACCUMULATE: VisibilityResultType = VisibilityResultType::Accumulate;

/// A word whose low half a count of 16 takes past 32 bits.
const CARRY: u64 = 0xffff_fff8;

/// A word that a count of 16 would take past 64 bits, were the count added to all of it.
const WRAP: u64 = u64::MAX - 7;

/// Every case of the family, in the order the report lists them.
const SHAPES: &[Shape] = &[
    // Probe 1's visibility_modes: the Boolean offsets, then a second pass that rewrites only its
    // own offset.
    Shape {
        name: "visibility-boolean-offsets",
        words: 8,
        passes: &[
            &[
                Act::Mode(BOOL, 0),
                Act::draw(),
                Act::Mode(BOOL, 24),
                Act::draw_nothing(),
                Act::Mode(BOOL, 40),
                Act::Mode(OFF, 48),
                Act::draw(),
                Act::Mode(BOOL, 56),
                Act::draw(),
                Act::draw(),
            ],
            &[Act::Mode(BOOL, 8), Act::draw()],
        ],
        depth: None,
        samples: 1,
        kind: RESET,
        start: None,
        joined: false,
    },
    // Probe 5's vis_revisit_boolean_then_none: offset 0 taken again after offset 8, whose first
    // draw covered and whose second did not.
    Shape {
        name: "visibility-boolean-revisit-after-covering",
        words: 2,
        passes: &[&[
            Act::Mode(BOOL, 0),
            Act::draw(),
            Act::Mode(BOOL, 8),
            Act::draw_nothing(),
            Act::Mode(BOOL, 0),
            Act::draw_nothing(),
        ]],
        depth: None,
        samples: 1,
        kind: RESET,
        start: None,
        joined: false,
    },
    // Probe 5's vis_revisit_none_then_boolean: the same three draws the other way round.
    Shape {
        name: "visibility-boolean-revisit-after-nothing",
        words: 2,
        passes: &[&[
            Act::Mode(BOOL, 0),
            Act::draw_nothing(),
            Act::Mode(BOOL, 8),
            Act::draw(),
            Act::Mode(BOOL, 0),
            Act::draw(),
        ]],
        depth: None,
        samples: 1,
        kind: RESET,
        start: None,
        joined: false,
    },
    // Probe 1's visibility_occluded: a draw at 0.5 behind the cleared depth 0.25 under `Less`
    // passes no sample, and one at 0.125 in front of it passes every one.
    Shape {
        name: "visibility-boolean-occluded",
        words: 2,
        passes: &[&[
            Act::Mode(BOOL, 8),
            Act::draw(),
            Act::Mode(BOOL, 0),
            Act::Draw {
                z: 0.125,
                covering: true,
                instances: 1,
            },
        ]],
        depth: Some(Depth {
            format: PixelFormat::Depth32Float,
            clear: 0.25,
            compare: CompareFunction::Less,
        }),
        samples: 1,
        kind: RESET,
        start: None,
        joined: false,
    },
    // Probe 5's vis_indirect_no_instances, as Boolean: an indirect draw of three vertices and no
    // instances.
    Shape {
        name: "visibility-boolean-indirect-no-instances",
        words: 1,
        passes: &[&[Act::Mode(BOOL, 0), Act::DrawIndirect]],
        depth: None,
        samples: 1,
        kind: RESET,
        start: None,
        joined: false,
    },
    // Probe 1's counts at their offsets: drawn, a draw covering nothing, two draws, three
    // instances, set with no draw; beside a `Boolean` offset, and a draw after `Disabled`.
    Shape {
        name: "visibility-counting-offsets",
        words: 8,
        passes: &[&[
            Act::Mode(COUNT, 0),
            Act::draw(),
            Act::Mode(COUNT, 8),
            Act::draw_nothing(),
            Act::Mode(COUNT, 16),
            Act::draw(),
            Act::draw(),
            Act::Mode(COUNT, 24),
            Act::Draw {
                z: DEPTH,
                covering: true,
                instances: 3,
            },
            Act::Mode(COUNT, 32),
            Act::Mode(BOOL, 40),
            Act::draw(),
            Act::Mode(OFF, 0),
            Act::draw(),
        ]],
        depth: None,
        samples: 1,
        kind: RESET,
        start: None,
        joined: false,
    },
    // Probe 1's visibility_occluded, counted.
    Shape {
        name: "visibility-counting-occluded",
        words: 2,
        passes: &[&[
            Act::Mode(COUNT, 8),
            Act::draw(),
            Act::Mode(COUNT, 0),
            Act::Draw {
                z: 0.125,
                covering: true,
                instances: 1,
            },
        ]],
        depth: Some(Depth {
            format: PixelFormat::Depth32Float,
            clear: 0.25,
            compare: CompareFunction::Less,
        }),
        samples: 1,
        kind: RESET,
        start: None,
        joined: false,
    },
    // Probe 5's vis_revisit_counting: offset 0 taken again after offset 8 sums its draws.
    Shape {
        name: "visibility-counting-revisit",
        words: 2,
        passes: &[&[
            Act::Mode(COUNT, 0),
            Act::draw(),
            Act::Mode(COUNT, 8),
            Act::draw(),
            Act::Mode(COUNT, 0),
            Act::draw(),
        ]],
        depth: None,
        samples: 1,
        kind: RESET,
        start: None,
        joined: false,
    },
    // Probe 5's vis_revisit_after_disabled: the same, with `Disabled` between.
    Shape {
        name: "visibility-counting-revisit-after-disabled",
        words: 2,
        passes: &[&[
            Act::Mode(COUNT, 0),
            Act::draw(),
            Act::Mode(OFF, 0),
            Act::draw(),
            Act::Mode(COUNT, 0),
            Act::draw(),
        ]],
        depth: None,
        samples: 1,
        kind: RESET,
        start: None,
        joined: false,
    },
    // Probe 1's 4x pass: the fullscreen triangle passes every sample of the 4 by 4 target, which
    // is 64 at 4x. The attachment is multisampled and private, and the pass does not store it.
    Shape {
        name: "visibility-counting-four-samples",
        words: 1,
        passes: &[&[Act::Mode(COUNT, 0), Act::draw()]],
        depth: None,
        samples: 4,
        kind: RESET,
        start: None,
        joined: false,
    },
    // `Accumulate`, counted, one command buffer a pass: a covering draw adds its 16 samples to
    // the word, a draw covering nothing adds none, an offset set with no draw keeps its word, and
    // the sum runs past 32 bits and past 64; a second pass adds to the first's words.
    Shape {
        name: "visibility-counting-accumulate",
        words: 5,
        passes: &[
            &[
                Act::Mode(COUNT, 0),
                Act::draw(),
                Act::Mode(COUNT, 8),
                Act::draw_nothing(),
                Act::Mode(COUNT, 16),
                Act::Mode(COUNT, 24),
                Act::draw(),
                Act::Mode(COUNT, 32),
                Act::draw(),
            ],
            &[Act::Mode(COUNT, 0), Act::draw(), Act::Mode(COUNT, 8)],
        ],
        depth: None,
        samples: 1,
        kind: ACCUMULATE,
        start: Some(&[100, 5, 7, CARRY, WRAP]),
        joined: false,
    },
    // The same two passes' first words in **one** command buffer.
    Shape {
        name: "visibility-counting-accumulate-one-command-buffer",
        words: 2,
        passes: &[
            &[Act::Mode(COUNT, 0), Act::draw(), Act::Mode(COUNT, 8)],
            &[Act::Mode(COUNT, 0), Act::draw()],
        ],
        depth: None,
        samples: 1,
        kind: ACCUMULATE,
        start: Some(&[100, 7]),
        joined: true,
    },
    // `Accumulate` of an offset taken again after another offset: both of its draws add to the
    // word.
    Shape {
        name: "visibility-counting-accumulate-revisit",
        words: 2,
        passes: &[&[
            Act::Mode(COUNT, 0),
            Act::draw(),
            Act::Mode(COUNT, 8),
            Act::draw(),
            Act::Mode(COUNT, 0),
            Act::draw(),
        ]],
        depth: None,
        samples: 1,
        kind: ACCUMULATE,
        start: Some(&[100, 5]),
        joined: false,
    },
    // `Accumulate`, `Boolean`: words 0, 2 and 3 under a covering draw, the same under a draw
    // covering nothing, and an offset set with no draw; then a second command buffer's covering
    // draw at the word that stayed 0.
    Shape {
        name: "visibility-boolean-accumulate",
        words: 7,
        passes: &[
            &[
                Act::Mode(BOOL, 0),
                Act::draw(),
                Act::Mode(BOOL, 8),
                Act::draw(),
                Act::Mode(BOOL, 16),
                Act::draw(),
                Act::Mode(BOOL, 24),
                Act::draw_nothing(),
                Act::Mode(BOOL, 32),
                Act::draw_nothing(),
                Act::Mode(BOOL, 40),
                Act::draw_nothing(),
                Act::Mode(BOOL, 48),
            ],
            &[Act::Mode(BOOL, 24), Act::draw()],
        ],
        depth: None,
        samples: 1,
        kind: ACCUMULATE,
        start: Some(&[0, 2, 3, 0, 2, 3, 9]),
        joined: false,
    },
];

/// The arguments an indirect draw of three vertices and no instances reads:
/// `MTLDrawPrimitivesIndirectArguments`, four 32-bit words.
const NO_INSTANCES: [u32; 4] = [3, 0, 0, 0];

/// One case of the family.
pub struct Visibility {
    shape: &'static Shape,
}

/// Every case of the family, in the order the report lists them.
pub fn all() -> Vec<Visibility> {
    SHAPES.iter().map(|shape| Visibility { shape }).collect()
}

/// The degrade notices a visibility pass may announce. As with every family, it is a list rather
/// than a device query, because a case never branches on the device: it asks by running and reads
/// what came back. A venue that announces a kind not here fails the case until its kind is added.
const PASS_NOTICES: &[Degradation] = &[
    // A `Shared` buffer the device keeps in system memory, which the visibility buffer is.
    Degradation::SharedInSystemMemory,
    // The multisampled attachment of a `MultisampleResolve` store, and a `DontCare` store, are
    // Vulkan's `DONT_CARE` on a driver measured to leave the earlier texels there. Nothing of
    // these cases reads a colour attachment -- the visibility result is the answer, and it is
    // compared exactly -- so naming the kind declares what our side announced and loosens no byte.
    Degradation::DontCareStore,
];

/// Each read's `Boolean` results, as (the pass's index, the result's offset): the offsets whose
/// last mode set, in that pass or one before it, is `Boolean`, which the pass's `words` site holds,
/// at each pass the words are read after ([`groups`]).
fn boolean_words(shape: &Shape) -> Vec<(usize, u64)> {
    let read: Vec<usize> = groups(shape)
        .iter()
        .filter_map(|g| g.last().copied())
        .collect();
    let mut held: BTreeSet<u64> = BTreeSet::new();
    let mut out = Vec::new();
    for (index, acts) in shape.passes.iter().enumerate() {
        for act in *acts {
            match act {
                Act::Mode(BOOL, offset) => {
                    held.insert(*offset);
                }
                Act::Mode(COUNT, offset) => {
                    held.remove(offset);
                }
                Act::Mode(..) | Act::Draw { .. } | Act::DrawIndirect => {}
            }
        }
        if read.contains(&index) {
            out.extend(held.iter().map(|&offset| (index, offset)));
        }
    }
    out
}

/// The shape's passes by command buffer, as indices: all in one where it is `joined`, else one
/// each.
fn groups(shape: &Shape) -> Vec<Vec<usize>> {
    let all = 0..shape.passes.len();
    match shape.joined {
        true => vec![all.collect()],
        false => all.map(|index| vec![index]).collect(),
    }
}

/// The bytes the shape's buffer starts as: its `start` words, or [`UNWRITTEN`] bytes.
fn start(shape: &Shape) -> Vec<u8> {
    match shape.start {
        Some(words) => words.iter().flat_map(|word| word.to_le_bytes()).collect(),
        None => vec![UNWRITTEN; 8 * shape.words],
    }
}

/// The eight bytes `rt6_full` reads at vertex buffer 0: the depth, then the slope, which every
/// shape here leaves at 0.
fn depth_bytes(z: f32) -> Vec<u8> {
    let mut bytes = z.to_le_bytes().to_vec();
    bytes.extend(0.0f32.to_le_bytes());
    bytes
}

/// The indirect arguments' bytes.
fn argument_bytes() -> Vec<u8> {
    NO_INSTANCES
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect()
}

/// Every depth a shape's draws take, once each, in the order they first appear: one vertex buffer
/// a depth is enough, since a buffer a draw reads is read at the draw.
fn depths(shape: &Shape) -> Vec<f32> {
    let mut found: Vec<f32> = Vec::new();
    let hold = |z: f32, found: &mut Vec<f32>| {
        if !found.iter().any(|held| held.to_bits() == z.to_bits()) {
            found.push(z);
        }
    };
    for pass in shape.passes {
        for act in *pass {
            match act {
                Act::Draw { z, .. } => hold(*z, &mut found),
                // An indirect draw reads the same vertex buffer a direct one does, at [`DEPTH`];
                // only its vertex and instance counts come from the arguments buffer.
                Act::DrawIndirect => hold(DEPTH, &mut found),
                Act::Mode(..) => {}
            }
        }
    }
    found
}

/// One parameter of a call, little-endian.
fn word(out: &mut Vec<u8>, value: u64) {
    out.extend(value.to_le_bytes());
}

/// The calls a shape's passes make, in order: each act keyed by **the name of the Metal call it
/// makes and the values handed that call**, with each pass opening on the encoder that runs it
/// and, under `Accumulate`, on its descriptor's `visibilityResultType`.
///
/// This replaced `format!("{:?}", shape.passes)`, which keyed a case by the printed text of
/// [`Act`]: that text moves on any change to the type, including one that changes no call these
/// cases make, and would stale every recording of the family at once with nothing to say so.
///
/// [`Act`] is this module's own type, so the match is **exhaustive with no `..`** — a field added
/// to one of its variants is a compile error here, and a new variant must name its own call. The
/// pass delimiter is a call too (`renderCommandEncoderWithDescriptor:`) rather than a bracket, so
/// the nesting is keyed by something Metal does.
///
/// The pipeline and the depth/stencil state each pass sets are not here: they are the case's own
/// `render-pipeline` and `depth-stencil-state` inputs, which key them in full.
fn act_words(shape: &Shape) -> Vec<u8> {
    let mut out = Vec::new();
    for acts in shape.passes {
        out.extend(b"renderCommandEncoderWithDescriptor");
        // The descriptor's type, only where it accumulates, so every other case keys as it always
        // did.
        if shape.kind != RESET {
            out.extend(crate::render::VISIBILITY_RESULT_TYPE.as_bytes());
            word(&mut out, shape.kind.raw() as u64);
        }
        for act in *acts {
            match act {
                Act::Mode(mode, offset) => {
                    out.extend(b"setVisibilityResultModeOffset");
                    word(&mut out, mode.raw());
                    word(&mut out, *offset);
                }
                Act::Draw {
                    z,
                    covering,
                    instances,
                } => {
                    // The draw reads the vertex buffer holding this depth. The bits name which of
                    // the case's own `buffer:0@<bits>` inputs that is, and those inputs carry the
                    // bytes; nothing of a side's handle is keyed.
                    out.extend(b"setVertexBufferOffsetAtIndex");
                    word(&mut out, u64::from(z.to_bits()));
                    word(&mut out, 0);
                    word(&mut out, 0);
                    out.extend(b"drawPrimitivesVertexStartVertexCountInstanceCount");
                    word(&mut out, u64::from(PrimitiveType::Triangle.as_raw()));
                    word(&mut out, if *covering { 0 } else { 3 });
                    word(&mut out, 3);
                    word(&mut out, u64::from(*instances));
                }
                Act::DrawIndirect => {
                    out.extend(b"setVertexBufferOffsetAtIndex");
                    word(&mut out, u64::from(DEPTH.to_bits()));
                    word(&mut out, 0);
                    word(&mut out, 0);
                    // The vertex and instance counts are the arguments buffer's, which the case
                    // states as its own `arguments` input, so the call itself takes the primitive
                    // type and the offset alone.
                    out.extend(b"drawPrimitivesIndirectBufferIndirectBufferOffset");
                    word(&mut out, u64::from(PrimitiveType::Triangle.as_raw()));
                    word(&mut out, 0);
                }
            }
        }
    }
    out
}

/// The `MTLDepthStencilDescriptor` the state is made from: the call's name and the one field this
/// family states.
///
/// This replaced `format!("{state:?}")`, which was the worse of the two `Debug` keys here and the
/// reason this packet exists: [`DepthStencilDescriptor`] is a **public type of the published
/// `metal2vulkan` crate**, so a field added to it in another crate staled these recordings, and
/// nothing in either crate connected the two.
///
/// It cannot take the protection the rest of this module's keys take. `DepthStencilDescriptor` and
/// `StencilDescriptor` are `#[non_exhaustive]`, so this crate **cannot** destructure them field by
/// field — the compiler requires a `..` — and there is no compile error to be had across that
/// boundary. The substitute is to state the fields this family sets and **prove the rest are
/// Metal's own defaults**: the descriptor is compared with `DepthStencilDescriptor::default()`
/// carrying only those fields, and one that differs anywhere else is refused rather than keyed. So
/// a field this family starts setting cannot be quietly left out of the key, while a field it
/// leaves at the default writes nothing — which is what keeps a widened descriptor from staling
/// these recordings at all.
fn depth_state_words(state: &DepthStencilDescriptor) -> Result<Vec<u8>, String> {
    let mut stated = DepthStencilDescriptor::default();
    stated.depth_compare_function = state.depth_compare_function;
    if *state != stated {
        return Err(
            "the depth/stencil state of this family names its depth compare function alone, and \
             this descriptor differs from Metal's defaults in some other field: name that field \
             here, or its recordings are keyed against a state they were not taken from"
                .to_owned(),
        );
    }
    let mut out = b"newDepthStencilStateWithDescriptor".to_vec();
    // `depthWriteEnabled` is off and both faces carry Metal's default stencil, which the
    // comparison above has just proved, so the compare function is the whole of it.
    out.extend(b"depthCompareFunction");
    word(&mut out, u64::from(state.depth_compare_function.as_raw()));
    Ok(out)
}

/// Whether any act of `shape` draws indirectly.
fn draws_indirect(shape: &Shape) -> bool {
    shape
        .passes
        .iter()
        .flat_map(|pass| pass.iter())
        .any(|act| matches!(act, Act::DrawIndirect))
}

impl Visibility {
    /// The fixture the case runs.
    fn fixture(&self) -> Fixture {
        Fixture::new(FIXTURE)
    }

    /// The one color state the pipeline holds.
    fn color_state(&self) -> [Option<ColorState>; 1] {
        [Some(ColorState::new(COLOR))]
    }

    /// The pipeline the case makes: the fullscreen triangle and white, over the color attachment,
    /// at the shape's sample count, naming the depth format where the shape has one.
    fn pipeline_spec<'a>(&self, colors: &'a [Option<ColorState>]) -> RenderPipelineSpec<'a> {
        RenderPipelineSpec::new(VERTEX, FRAGMENT, colors)
            .multisampled(self.shape.samples)
            .depth_stencil(self.shape.depth.map(|depth| depth.format), None)
    }

    /// The depth and stencil state the shape takes: its compare function, no depth write, and
    /// Metal's default stencil on both faces.
    fn depth_state(&self) -> Option<DepthStencilDescriptor> {
        self.shape.depth.map(|depth| {
            let mut state = DepthStencilDescriptor::default();
            state.depth_compare_function = depth.compare;
            state.depth_write_enabled = false;
            state
        })
    }

    /// The depth attachment one pass takes for `depth`, exactly as the pass encodes it: a 4 by 4
    /// texture of the shape's format, cleared to the shape's depth and not stored.
    ///
    /// One function, read by both the pass and the input fingerprint, so a change to the load or
    /// the store action moves the key as it must instead of leaving a recording matching a program
    /// it was not taken from. `PassDepth::words` fingerprints nothing of the handle, so the key may
    /// pass any.
    fn pass_depth(&self, texture: Handle, depth: &Depth) -> PassDepth {
        PassDepth::cleared(texture, depth.clear).stored(StoreAction::DontCare)
    }

    /// The depth attachment's own bytes: the texture it is over, and the actions the pass takes on
    /// it.
    ///
    /// `Depth` is this module's own type and is destructured with **no `..`**, deliberately: a
    /// field added to it is then a compile error here, where the `format!("{:?}")` this replaced
    /// would have changed the key for a field no call takes and staled both recordings. The
    /// texture is keyed in the same encoding the colour `target` input uses rather than in one
    /// invented here, and the actions come from [`Visibility::pass_depth`], which the pass itself
    /// reads.
    fn depth_attachment_words(&self, depth: &Depth) -> Vec<u8> {
        let Depth {
            format,
            clear: _,
            // The clear depth is in `pass_depth`'s own words below, and the compare function
            // belongs to the depth/stencil state, which `depth_state_words` keys. Both are named
            // here so that adding a field to `Depth` still stops this compiling.
            compare: _,
        } = depth;
        let mut out = b"depthAttachmentTexture".to_vec();
        out.extend(TargetSpec::plain(*format, [SIDE, SIDE]).words());
        out.extend(b"depthAttachmentActions");
        out.extend(self.pass_depth(Handle(0), depth).words());
        out
    }

    /// The target the shape's passes render into: a 4 by 4 `RGBA8Unorm` render target, shared and
    /// stored at 1x, private and discarded at 4x, because Metal stores no multisampled texture a
    /// host reads.
    fn target_spec(&self) -> TargetSpec {
        let spec = TargetSpec::plain(COLOR, [SIDE, SIDE]);
        match self.shape.samples {
            1 => spec.shared(),
            samples => spec.multisampled(samples),
        }
    }
}

impl Case for Visibility {
    fn name(&self) -> &'static str {
        self.shape.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // The notices site, and each `Boolean` result's word by its contract (the module docs): a
        // visibility result is a number Metal wrote, and any other number that differs is a bug,
        // not a near-equivalent. Nothing else here is loosened.
        let mut all: Vec<Allowance> = PASS_NOTICES
            .iter()
            .map(|kind| {
                Allowance::new(
                    format!("case/{NOTICES_SITE}/*"),
                    Loosening::Unchecked,
                    Justification::notice(*kind),
                )
            })
            .collect();
        let booleans = boolean_words(self.shape);
        if !booleans.is_empty() {
            all.push(crate::replay::notice_allowance(
                Degradation::DeviceBooleanVisibility.code(),
            ));
        }
        for (index, offset) in booleans {
            all.push(Allowance::new(
                format!("case/pass-{index}/words/{offset}"),
                Loosening::Nonzero,
                Justification::notice(Degradation::DeviceBooleanVisibility),
            ));
        }
        all
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let fixture = self.fixture();
        [RUNTIME_TEXT, METAL_TEXT]
            .into_iter()
            .map(|extension| {
                Ok(ShaderRef {
                    fixture: format!("{}{extension}", fixture.name),
                    entry: format!("{VERTEX}+{FRAGMENT}"),
                    source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
                })
            })
            .collect()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = self.fixture();
        let colors = self.color_state();
        let mut inputs = vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("render-pipeline", &self.pipeline_spec(&colors).words()),
            Input::bytes("target", &self.target_spec().words()),
            Input::bytes("visibility", &start(self.shape)),
            Input::bytes("acts", &act_words(self.shape)),
        ];
        // Named only where a shape runs its passes in one command buffer, so every other case keys
        // as it always did.
        if self.shape.joined {
            let mut words = b"commandBuffer".to_vec();
            word(&mut words, self.shape.passes.len() as u64);
            inputs.push(Input::bytes("command-buffer", &words));
        }
        if let Some(state) = self.depth_state() {
            inputs.push(Input::bytes(
                "depth-stencil-state",
                &depth_state_words(&state)?,
            ));
        }
        if let Some(depth) = self.shape.depth {
            inputs.push(Input::bytes(
                "depth-attachment",
                &self.depth_attachment_words(&depth),
            ));
        }
        for z in depths(self.shape) {
            inputs.push(Input::bytes(
                format!("buffer:0@{}", z.to_bits()),
                &depth_bytes(z),
            ));
        }
        if draws_indirect(self.shape) {
            inputs.push(Input::bytes("arguments", &argument_bytes()));
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let loaded = self.fixture().loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = crate::case::made(sites, "library", side.library(&loaded.source()))?;
        let colors = self.color_state();
        let state = crate::case::made(
            sites,
            "render-pipeline",
            side.render_pipeline(library, &self.pipeline_spec(&colors)),
        )?;
        let depth_state = match self.depth_state() {
            Some(spec) => Some(crate::case::made(
                sites,
                "depth-stencil-state",
                side.depth_stencil_state(&spec),
            )?),
            None => None,
        };
        // One vertex buffer a depth, one indirect-arguments buffer where a shape draws indirectly,
        // and one visibility buffer every pass of the case writes into.
        let mut vertex: Vec<(u32, Handle)> = Vec::new();
        for z in depths(self.shape) {
            let handle = crate::case::made(
                sites,
                &format!("buffer:0@{}", z.to_bits()),
                side.buffer(&depth_bytes(z)),
            )?;
            vertex.push((z.to_bits(), handle));
        }
        let arguments = match draws_indirect(self.shape) {
            true => Some(crate::case::made(
                sites,
                "arguments",
                side.buffer(&argument_bytes()),
            )?),
            false => None,
        };
        let visibility = crate::case::made(sites, "visibility", side.buffer(&start(self.shape)))?;
        let length = 8 * self.shape.words;
        let held = Held {
            state,
            depth_state,
            vertex: &vertex,
            arguments,
            visibility,
        };
        for group in groups(self.shape) {
            let mut work = Vec::with_capacity(group.len());
            for &index in &group {
                work.push(RenderWork::Pass(self.pass(side, sites, index, &held)?));
            }
            // The words are read after the command buffer, at its last pass's site.
            let last = group.last().copied().unwrap_or_default();
            let at = format!("pass-{last}");
            let stop = |at: String, failure: Failure| Abort { at, failure };
            let outcome = side
                .render(&work)
                .map_err(|failure| stop(format!("{at}/render"), failure))?;
            let mut node = Node::branch(at.clone(), Vec::new());
            let mut calls = Node::branch("calls", Vec::new());
            for (piece, refused) in outcome.refused.iter().enumerate() {
                calls = calls.with(Node::leaf(
                    format!("{piece}-render"),
                    Observed::value(match refused {
                        Some(failure) => failure.recorded(),
                        None => Value::ok(Value::Absent),
                    }),
                ));
            }
            node = node.with(calls);
            if let Some((piece, failure)) = outcome
                .refused
                .iter()
                .enumerate()
                .find_map(|(piece, refused)| refused.as_ref().map(|f| (piece, f)))
            {
                sites.push(node);
                return Err(Abort {
                    at: format!("{at}/calls/{piece}-render"),
                    failure: failure.clone(),
                });
            }
            node = node.with(Node::leaf(
                "status",
                Observed::value(Value::ok(outcome.status.clone())),
            ));
            node = node.with(Node::leaf(
                "command-buffer-error",
                Observed::value(match &outcome.error {
                    Some(error) => Value::Variant {
                        name: "Some".to_owned(),
                        payload: Box::new(error.clone()),
                    },
                    None => Value::unit_variant("None"),
                }),
            ));
            node = node.with(Node::leaf(
                "words",
                Observed::bytes(
                    side.read_buffer(visibility, 0, length)
                        .map_err(|failure| stop(format!("{at}/words"), failure))?,
                    ByteLayout::buffer(0, length as u64, Element::U64),
                ),
            ));
            sites.push(node);
        }
        Ok(())
    }
}

/// What every pass of a case draws with, made once before them.
struct Held<'a> {
    state: Handle,
    depth_state: Option<Handle>,
    /// One vertex buffer a depth, by the depth's bits.
    vertex: &'a [(u32, Handle)],
    arguments: Option<Handle>,
    visibility: Handle,
}

impl Visibility {
    /// Pass `index` of the shape: a fresh target, as the lane makes one a pass, so no pass reads
    /// what another left in the color attachment, its acts as steps, and its depth attachment
    /// where the shape has one.
    fn pass(
        &self,
        side: &mut dyn Side,
        sites: &mut Sites,
        index: usize,
        held: &Held<'_>,
    ) -> Result<RenderPass, Abort> {
        let at = format!("pass-{index}");
        let stop = |at: String, failure: Failure| Abort { at, failure };
        let target = crate::case::made(
            sites,
            &format!("{at}/target"),
            side.target(&self.target_spec()),
        )?;
        let mut steps = vec![RenderStep::Pipeline(held.state)];
        if let Some(state) = held.depth_state {
            steps.push(RenderStep::DepthStencilState(state));
        }
        for act in self.shape.passes[index] {
            match act {
                Act::Mode(mode, offset) => steps.push(RenderStep::VisibilityResultMode {
                    mode: *mode,
                    offset: *offset,
                }),
                Act::Draw {
                    z,
                    covering,
                    instances,
                } => {
                    let depth = held
                        .vertex
                        .iter()
                        .find(|(bits, _)| *bits == z.to_bits())
                        .map(|(_, handle)| *handle)
                        .ok_or_else(|| {
                            stop(at.clone(), Failure::named("NoDepthBuffer", format!("{z}")))
                        })?;
                    steps.push(RenderStep::vertex_buffer(depth, 0));
                    steps.push(RenderStep::Draw {
                        primitive: PrimitiveType::Triangle,
                        first: if *covering { 0 } else { 3 },
                        count: 3,
                        instances: *instances,
                    });
                }
                Act::DrawIndirect => {
                    let arguments = held.arguments.ok_or_else(|| {
                        stop(
                            at.clone(),
                            Failure::named("NoArguments", "an indirect draw"),
                        )
                    })?;
                    let depth = held
                        .vertex
                        .iter()
                        .find(|(bits, _)| *bits == DEPTH.to_bits())
                        .map(|(_, handle)| *handle)
                        .ok_or_else(|| {
                            stop(
                                at.clone(),
                                Failure::named("NoDepthBuffer", "an indirect draw"),
                            )
                        })?;
                    steps.push(RenderStep::vertex_buffer(depth, 0));
                    steps.push(RenderStep::DrawIndirect {
                        primitive: PrimitiveType::Triangle,
                        arguments,
                        offset: 0,
                    });
                }
            }
        }
        // At 4x the attachment is private and the pass does not store it: the question is the
        // samples the draws passed, not the texels.
        let color = match self.shape.samples {
            1 => PassColor::cleared(target, [0.0; 4]),
            _ => PassColor::cleared(target, [0.0; 4]).stored(StoreAction::DontCare),
        };
        let mut pass = RenderPass::clearing(vec![color])
            .drawing(steps)
            .counting_into(held.visibility)
            .typed(self.shape.kind);
        if let Some(depth) = self.shape.depth {
            let attachment = crate::case::made(
                sites,
                &format!("{at}/depth"),
                side.target(&TargetSpec::plain(depth.format, [SIDE, SIDE])),
            )?;
            pass = pass.with_depth(self.pass_depth(attachment, &depth));
        }
        Ok(pass)
    }
}

#[cfg(test)]
mod tests {
    //! The cross-check the port owes the lane: the words Metal wrote, as the recordings hold them,
    //! against the lane's table.
    //!
    //! The table is the lane's (`runtime_lane/visibility.rs`), restated here because a conformance
    //! case may not depend on the validation crate. Its comment says it was measured on an **M4**;
    //! this recording is of an M3 Max. A disagreement is therefore a fact about one of the two
    //! chips, and the test names the case and the word.

    use super::*;
    use crate::recording::Recording;
    use crate::transcript::Transcript;

    /// A word Metal never wrote, as a 64-bit value.
    const U: u64 = u64::from_le_bytes([UNWRITTEN; 8]);

    /// The lane's words, by case: what Metal wrote after each command buffer, in order.
    const TABLE: &[(&str, &[&[u64]])] = &[
        (
            "visibility-boolean-offsets",
            &[&[1, U, U, 0, U, 0, U, 1], &[1, 1, U, 0, U, 0, U, 1]],
        ),
        ("visibility-boolean-revisit-after-covering", &[&[1, 0]]),
        ("visibility-boolean-revisit-after-nothing", &[&[1, 1]]),
        ("visibility-boolean-occluded", &[&[1, 0]]),
        ("visibility-boolean-indirect-no-instances", &[&[0]]),
        (
            "visibility-counting-offsets",
            &[&[16, 0, 32, 48, 0, 1, U, U]],
        ),
        ("visibility-counting-occluded", &[&[16, 0]]),
        ("visibility-counting-revisit", &[&[32, 16]]),
        ("visibility-counting-revisit-after-disabled", &[&[32, U]]),
        ("visibility-counting-four-samples", &[&[64]]),
        // The `Accumulate` rows, the lane's too: a count is added to the word's low 32 bits,
        // which wrap, and its high 32 bits stay.
        (
            "visibility-counting-accumulate",
            &[
                &[116, 5, 7, 8, 0xffff_ffff_0000_0008],
                &[132, 5, 7, 8, 0xffff_ffff_0000_0008],
            ],
        ),
        (
            "visibility-counting-accumulate-one-command-buffer",
            &[&[132, 7]],
        ),
        ("visibility-counting-accumulate-revisit", &[&[132, 21]]),
        (
            "visibility-boolean-accumulate",
            &[&[1, 3, 3, 0, 2, 3, 9], &[1, 3, 3, 1, 2, 3, 9]],
        ),
    ];

    /// The one outcome of `name`'s recorded Metal answer.
    fn only_outcome(name: &str) -> Transcript {
        let recording: Recording = crate::recording::read(name)
            .unwrap_or_else(|why| panic!("{name}: no recording ({why})"));
        let answer = recording
            .answers
            .first()
            .unwrap_or_else(|| panic!("{name}: no answer"));
        assert_eq!(answer.outcomes.len(), 1, "{name}: more than one outcome");
        answer.outcomes[0].transcript.clone()
    }

    /// The 64-bit words the site at `path` of `case`'s `transcript` holds.
    fn words(case: &str, transcript: &Transcript, path: &str) -> Vec<u64> {
        let mut node = &transcript.root;
        for name in path.trim_start_matches("case/").split('/') {
            node = node
                .child(name)
                .unwrap_or_else(|| panic!("{case}: no site {path} (stopped at {name})"));
        }
        let Value::Bytes(bytes) = &node.observed.value else {
            panic!("{case}: {path} holds no bytes");
        };
        bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|word| u64::from_le_bytes(*word))
            .collect()
    }

    /// Metal's own visibility results on this M3 Max are the lane's words, pass by pass.
    ///
    /// Every case the table names must have a recording, so a case that was never recorded fails
    /// here rather than being skipped: the table is the claim, and a claim with no run behind it
    /// is the thing this suite exists to stop.
    #[test]
    fn the_recorded_results_are_the_lanes_words() {
        for (name, passes) in TABLE {
            let shape = SHAPES
                .iter()
                .find(|shape| shape.name == *name)
                .unwrap_or_else(|| {
                    panic!("{name}: the table names a case the family does not hold")
                });
            // One row a command buffer, read at its last pass.
            let reads: Vec<usize> = groups(shape)
                .iter()
                .filter_map(|g| g.last().copied())
                .collect();
            assert_eq!(reads.len(), passes.len(), "{name}: one row a read");
            let recorded = only_outcome(name);
            for (&index, want) in reads.iter().zip(passes.iter()) {
                assert_eq!(
                    words(name, &recorded, &format!("case/pass-{index}/words")),
                    **want,
                    "{name}: pass {index}"
                );
            }
        }
    }

    /// Every case of the family is in the table, states its inputs, and loosens **only** its
    /// notices site and its `Boolean` results' words, each by its contract alone.
    ///
    /// A visibility result is a number Metal wrote; a number that differs is a bug, never a
    /// near-equivalent, but that a `Boolean` result's nonzero value is the device's. A case that
    /// ever loosened another word, or a `Boolean` one by more than its contract, would fail here.
    #[test]
    fn every_case_is_tabled_and_loosens_its_notices_and_boolean_words_alone() {
        for case in all() {
            assert!(
                TABLE.iter().any(|(name, _)| *name == case.name()),
                "{}: no row in the lane's table",
                case.name()
            );
            let inputs = case.inputs().unwrap();
            assert!(
                inputs.iter().any(|input| input.name == "visibility"),
                "{}: no visibility buffer in its inputs",
                case.name()
            );
            assert_eq!(case.metal_source(), MetalSource::Run);
            case.shaders().unwrap();
            let words: Vec<String> = boolean_words(case.shape)
                .iter()
                .map(|(index, offset)| format!("case/pass-{index}/words/{offset}"))
                .collect();
            for allowance in case.allowances() {
                if words.contains(&allowance.path) {
                    assert_eq!(allowance.loosening, Loosening::Nonzero, "{}", case.name());
                    assert_eq!(
                        allowance.justified_by,
                        Justification::notice(Degradation::DeviceBooleanVisibility),
                        "{}",
                        case.name()
                    );
                    continue;
                }
                assert_eq!(
                    allowance.path,
                    format!("case/{NOTICES_SITE}/*"),
                    "{}: an allowance outside the notices site and the Boolean words",
                    case.name()
                );
                match &allowance.justified_by {
                    Justification::Degradation(code) => assert!(
                        PASS_NOTICES
                            .iter()
                            .chain([&Degradation::DeviceBooleanVisibility])
                            .any(|kind| kind.code() == code),
                        "{}: {code} is not a visibility-pass notice",
                        case.name()
                    ),
                    other => panic!("{}: an unexpected tolerance: {other:?}", case.name()),
                }
            }
        }
    }

    /// **Every byte of the `acts` key is a Metal call's name and the values it takes.** Each pass
    /// opens on the encoder that runs it and each act on the call it makes, so nothing here is
    /// keyed by a position or by the printed text of [`Act`] — which is what staled nothing yet
    /// only because no stream had touched the type.
    #[test]
    fn the_acts_key_names_the_metal_calls_and_holds_no_debug_text() {
        for shape in SHAPES {
            let words = act_words(shape);
            assert!(
                words.starts_with(b"renderCommandEncoderWithDescriptor"),
                "{}: the key does not open on the pass's encoder",
                shape.name
            );
            // A derived `Debug` of `&[&[Act]]` prints braces, field names and the variants' own
            // spellings. None of them can be in a key of call names and little-endian numbers.
            for print in [
                &b" { "[..],
                b"Mode(",
                b"Draw {",
                b"covering",
                b"instances",
                b"DrawIndirect",
            ] {
                assert!(
                    !words.windows(print.len()).any(|window| window == print),
                    "{}: the key holds {:?}",
                    shape.name,
                    String::from_utf8_lossy(print)
                );
            }
            // One `renderCommandEncoderWithDescriptor` a pass, so the nesting really is keyed.
            let passes = words
                .windows(b"renderCommandEncoderWithDescriptor".len())
                .filter(|window| *window == b"renderCommandEncoderWithDescriptor")
                .count();
            assert_eq!(passes, shape.passes.len(), "{}", shape.name);
        }
    }

    /// No two shapes key alike, and a shape's acts key moves when its program does. Without this
    /// the whole re-key could have collapsed ten cases onto one fingerprint and every recording
    /// would still have matched something.
    #[test]
    fn the_acts_key_tells_the_shapes_apart() {
        let keyed: Vec<(&str, Vec<u8>)> = SHAPES
            .iter()
            .map(|shape| (shape.name, act_words(shape)))
            .collect();
        for (at, (name, words)) in keyed.iter().enumerate() {
            for (other, theirs) in &keyed[at + 1..] {
                assert_ne!(words, theirs, "{name} keys as {other}");
            }
        }
        // And the three things an act carries each move it: the mode, the offset, and whether the
        // draw covers anything. Each program is a `const` because a shape's passes are `'static`.
        const BASE: &[&[Act]] = &[&[Act::Mode(BOOL, 0), Act::draw()]];
        const MODE: &[&[Act]] = &[&[Act::Mode(COUNT, 0), Act::draw()]];
        const OFFSET: &[&[Act]] = &[&[Act::Mode(BOOL, 8), Act::draw()]];
        const NOTHING: &[&[Act]] = &[&[Act::Mode(BOOL, 0), Act::draw_nothing()]];
        // Two passes are not one pass of the same acts: the encoder delimiter is what says so.
        const SPLIT: &[&[Act]] = &[&[Act::Mode(BOOL, 0)], &[Act::draw()]];
        let one = |acts: &'static [&'static [Act]]| {
            act_words(&Shape {
                name: "probe",
                words: 1,
                passes: acts,
                depth: None,
                samples: 1,
                kind: RESET,
                start: None,
                joined: false,
            })
        };
        let base = one(BASE);
        for (named, other) in [
            ("the mode", MODE),
            ("the offset", OFFSET),
            ("whether the draw covers", NOTHING),
            ("the pass boundary", SPLIT),
        ] {
            assert_ne!(base, one(other), "{named} does not move the key");
        }
    }

    /// **The depth/stencil key names the compare function alone, and refuses a descriptor that
    /// differs from Metal's defaults anywhere else.**
    ///
    /// This is the substitute for the protection the rest of this module's keys get from the
    /// compiler. `DepthStencilDescriptor` is `#[non_exhaustive]` and belongs to another crate, so
    /// no exhaustive destructure is possible here and a field added to it cannot be a compile
    /// error. The refusal is what stands in for one: a field this family starts setting shows up
    /// as an error from `inputs()`, not as a key quietly missing it.
    #[test]
    fn the_depth_state_key_names_the_compare_function_alone() {
        let mut state = DepthStencilDescriptor::default();
        state.depth_compare_function = CompareFunction::Less;
        let mut expected = b"newDepthStencilStateWithDescriptor".to_vec();
        expected.extend(b"depthCompareFunction");
        expected.extend(u64::from(CompareFunction::Less.as_raw()).to_le_bytes());
        assert_eq!(depth_state_words(&state).unwrap(), expected);
        // The compare function is in it, so two states do not key alike.
        let mut greater = state;
        greater.depth_compare_function = CompareFunction::Greater;
        assert_ne!(depth_state_words(&greater).unwrap(), expected);
        // And any other field off Metal's default is refused rather than keyed.
        let mut written = state;
        written.depth_write_enabled = true;
        assert!(depth_state_words(&written).is_err(), "a depth write");
        let mut masked = state;
        masked.front_face_stencil.write_mask = 0;
        assert!(depth_state_words(&masked).is_err(), "a front stencil mask");
        let mut back = state;
        back.back_face_stencil.stencil_compare_function = CompareFunction::Never;
        assert!(depth_state_words(&back).is_err(), "a back stencil compare");
        // Every state this family in fact builds is one the key states.
        for shape in SHAPES {
            let case = Visibility { shape };
            if let Some(held) = case.depth_state() {
                depth_state_words(&held).unwrap_or_else(|why| panic!("{}: {why}", shape.name));
            }
        }
    }

    /// The attachment key holds the texture's format and the actions the pass takes on it, so a
    /// recording cannot be read as one of another format or another store action.
    #[test]
    fn the_depth_attachment_key_holds_its_format_and_its_actions() {
        let depth = Depth {
            format: PixelFormat::Depth32Float,
            clear: 0.25,
            compare: CompareFunction::Less,
        };
        let case = Visibility { shape: &SHAPES[0] };
        let words = case.depth_attachment_words(&depth);
        let other = Depth {
            format: PixelFormat::Depth16Unorm,
            ..depth
        };
        assert_ne!(words, case.depth_attachment_words(&other), "the format");
        let cleared = Depth {
            clear: 0.5,
            ..depth
        };
        assert_ne!(words, case.depth_attachment_words(&cleared), "the clear");
        // The texture is keyed in the same encoding the colour `target` input uses.
        let texture = TargetSpec::plain(PixelFormat::Depth32Float, [SIDE, SIDE]).words();
        assert!(
            words.windows(texture.len()).any(|window| window == texture),
            "the attachment's texture is not keyed as a target"
        );
        // And the pass's own actions are the ones the pass encodes, from the one function both
        // read, so the key cannot drift from the program.
        let actions = case.pass_depth(Handle(0), &depth).words();
        assert!(
            words.windows(actions.len()).any(|window| window == actions),
            "the attachment's actions are not the pass's"
        );
    }

    /// Only the two occluded shapes carry a depth attachment, which is what makes their two inputs
    /// the one place a `DepthStencilDescriptor` reached a fingerprint at all.
    #[test]
    fn the_two_occluded_shapes_are_the_ones_with_a_depth_attachment() {
        let named: Vec<&str> = SHAPES
            .iter()
            .filter(|shape| shape.depth.is_some())
            .map(|shape| shape.name)
            .collect();
        assert_eq!(
            named,
            vec![
                "visibility-boolean-occluded",
                "visibility-counting-occluded"
            ]
        );
        for case in all() {
            let inputs = case.inputs().unwrap();
            let depth_inputs = inputs
                .iter()
                .filter(|input| input.name.starts_with("depth-"))
                .count();
            let wants = usize::from(case.shape.depth.is_some()) * 2;
            assert_eq!(depth_inputs, wants, "{}", case.name());
        }
    }
}
