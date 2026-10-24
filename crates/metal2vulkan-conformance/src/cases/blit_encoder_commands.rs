//! The blit encoder's commands no other case encodes: the two un-optioned copy selectors, the two
//! whole-texture copies, the mipmap filter, the four layout hints, the two host
//! synchronizations and a fence between two of one command buffer's encoders.
//!
//! `MTLBlitCommandEncoder` has twenty-two implemented items and the suite reached five of them: a
//! region copy each way, a texture-to-texture region copy, a buffer fill and a buffer copy. The
//! nine cases here drive thirteen more. Each is read the same way the family's existing cases are
//! read: **every byte of the read-back, the bytes a copy must not touch included**. A destination
//! buffer is made holding one guard byte, `0x7e`, and each copy is given a slot wider than its
//! region, so a copy that ran long, landed at the wrong offset or wrote a row's padding shows up
//! as a guard byte that moved rather than as a value nobody looked at.
//!
//! - [`whole_texture_copy`]: `copyFromTexture:toTexture:` between two 4 by 4 `R32Uint` arrays of
//!   three levels and two slices, the destination written first so every byte compared is a byte
//!   the case wrote. It pins **which levels and which slices** that selector pairs, which is the
//!   whole content of a call that names no region at all.
//! - [`texture_slice_counts`]: the same pair under
//!   `copyFromTexture:sourceSlice:sourceLevel:toTexture:destinationSlice:destinationLevel:
//!   sliceCount:levelCount:`, two slices of two levels from level 1 — so level 0 of both slices
//!   must still read what the case wrote into it and levels 1 and 2 must read the source's.
//! - [`texture_counts_of_none`]: the same call with `sliceCount` 0 and with `levelCount` 0. Plain
//!   Metal copies nothing for either, so the row's answer is a negative fact and the destination's
//!   own bytes, and a side that moved a texel emits bytes where Metal emits the fact.
//! - [`generated_mipmaps`]: `generateMipmapsForTexture:` over an 8 by 8 `RGBA8Unorm` texture of
//!   four levels, every level read back. **The one case here whose answer is a filter's rather
//!   than a move's**, so it is its own case: an implementation whose box filter rounds otherwise,
//!   or weights the corners otherwise, diverges here and in none of the others.
//! - [`optimized_contents`]: the four layout hints —
//!   `optimizeContentsForCPUAccess:`, `optimizeContentsForGPUAccess:slice:level:`,
//!   `optimizeContentsForCPUAccess:slice:level:` and `optimizeContentsForGPUAccess:` — encoded
//!   between a copy in and a copy out **of one command buffer**, so the bytes read back are the
//!   bytes those four hints were asked about and not a second run's.
//! - [`synchronized_managed_buffer`]: `synchronizeResource:` of a `Managed` buffer, read by the
//!   host through `contents`. This is the one case of the family whose call is **load-bearing on
//!   our own leg**: the runtime keeps a managed buffer's host copy apart from the device's, and
//!   nothing but this call copies the device's writes back into it. The buffer is given a known
//!   byte and synchronized once before the fill the case is about, so the bytes outside that fill
//!   are bytes the case wrote. It is the one case of the family that **announces** anything, and
//!   the only one that declares a notice, and the run that measured it is written at the
//!   declaration's own constant.
//! - [`synchronized_managed_texture`]: `synchronizeTexture:slice:level:` of a `Managed` texture,
//!   read by the host with `getBytes:` at a pitch six times its row — so the answer holds the
//!   padding between the rows and after the last one, which no device write may touch.
//! - [`unoptioned_copies`]: the two selectors that take **no** `MTLBlitOption`, beside the two
//!   that do, over the same bytes in one command buffer. Our runtime binds one method for each
//!   pair (Apple's header defines the shorter selector as the longer one with
//!   `MTLBlitOptionNone`), so what this case can still catch is **Metal's** two selectors
//!   disagreeing, and that is the honest claim: it exercises the item and holds Metal to its own
//!   equivalence, and it cannot fail on our side alone.
//! - [`fenced_encoders`]: `updateFence:` on one blit encoder of a command buffer and
//!   `waitForFence:` on the next, with the copy in before the fence and the copy out after it.
//!   Stated plainly: **Metal orders a command buffer's encoders anyway**, so the fence is
//!   redundant here and this case does not prove it orders anything. What it pins is that
//!   encoding the pair changes none of the bytes, which is what a consumer that encodes fences
//!   defensively needs, and it is what makes the two items exercised rather than named.
//!
//! # Why every case here is safe
//!
//! Every command is a blit the program's own resources are the operands of, every region lies
//! inside its level and slice (`every_region_lies_inside_its_texture`), there is no shader, no
//! draw, no dispatch and no loop, and every wait is bounded. The mipmap case names an
//! uncompressed, linearly filterable colour format, which is the shape `generateMipmapsForTexture:`
//! is defined for. Nothing allocates more than a few kibibytes.

use crate::allowance::Allowance;
use crate::case::{made, Abort, Case, Failure, Input, MetalSource, Side};
use crate::transcript::{ByteLayout, Element, Negative, Node, Observed, Sites, Value};
use crate::work::{Plane, Rect, Resource, Step, Storage, TextureSpec, Work};
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan_runtime::Degradation;

/// The byte every read-back buffer is made holding, so the bytes no copy wrote are a value the
/// case put there rather than whatever the allocator left.
const GUARD: u8 = 0x7e;

/// The byte a destination texture is written with before the copy the case is about, so a texel
/// the copy left alone is a recorded value too.
const BEFORE: u8 = 0xcd;

/// Bytes between one read-back slot's start and the next: wider than the widest region any case
/// reads into one, so each slot carries its own padding after the region.
const SLOT: usize = 320;

/// Bytes of guard before the first slot, so a copy that landed short of its offset shows.
const LEAD: usize = 32;

/// `length` bytes, byte `i` holding `i % 251`: a period no power of two divides, so a copy that
/// lands at the wrong offset shows in the values and not only in the guard.
fn counting(length: usize) -> Vec<u8> {
    (0..length).map(|index| (index % 251) as u8).collect()
}

/// The read-back buffer for `slots` slots: guard throughout, with the lead-in before the first.
fn read_back(slots: usize) -> Vec<u8> {
    vec![GUARD; LEAD + slots * SLOT]
}

/// Where slot `at` of a read-back buffer starts.
fn slot_at(at: usize) -> u64 {
    (LEAD + at * SLOT) as u64
}

/// The levels of a texture `width` by `width`, as their width and their bytes a texel: level `l`
/// is `width >> l` wide, down to one texel.
fn level_widths(width: u64, levels: u64) -> Vec<u64> {
    (0..levels).map(|level| (width >> level).max(1)).collect()
}

/// The 4 by 4 `R32Uint` array of three levels and two slices the texture-copy cases use. Both the
/// source and the destination are one of these, so every level pairs with one of its own size and
/// the copies the cases make are the shapes those selectors are defined for.
fn array_spec() -> TextureSpec {
    TextureSpec::texture_2d(PixelFormat::R32Uint.as_raw(), ARRAY_WIDTH, ARRAY_WIDTH)
        .levels(ARRAY_LEVELS)
        .array(ARRAY_SLICES)
        .used(TextureUsage::SHADER_READ.as_raw())
}

/// The array textures' extent, levels and slices.
const ARRAY_WIDTH: u64 = 4;
const ARRAY_LEVELS: u64 = 3;
const ARRAY_SLICES: u64 = 2;

/// Bytes one texel of the array textures takes.
const ARRAY_TEXEL: u64 = 4;

/// Each (slice, level) of the array textures, in the order the programs write and read them, with
/// the level's width and the bytes it holds.
fn array_regions() -> Vec<(u64, u64, u64, u64)> {
    let widths = level_widths(ARRAY_WIDTH, ARRAY_LEVELS);
    let mut out = Vec::new();
    for slice in 0..ARRAY_SLICES {
        for (level, width) in widths.iter().enumerate() {
            out.push((slice, level as u64, *width, width * width * ARRAY_TEXEL));
        }
    }
    out
}

/// The bytes a buffer holds that writes every region of an array texture, packed in
/// [`array_regions`]'s order with each level's rows tight.
fn array_source(byte: Option<u8>) -> Vec<u8> {
    let total: u64 = array_regions().iter().map(|(_, _, _, bytes)| bytes).sum();
    match byte {
        Some(byte) => vec![byte; total as usize],
        None => counting(total as usize),
    }
}

/// Where the region at `index` of [`array_regions`] starts in an [`array_source`] buffer.
fn array_offset(index: usize) -> u64 {
    array_regions()
        .iter()
        .take(index)
        .map(|(_, _, _, bytes)| bytes)
        .sum()
}

/// Write every region of `texture` from `buffer`, which holds them in [`array_regions`]'s order.
fn write_array(work: &mut Work, buffer: usize, texture: usize) {
    for (at, (slice, level, width, _)) in array_regions().into_iter().enumerate() {
        work.step(Step::CopyBufferToTexture {
            buffer,
            offset: array_offset(at),
            bytes_per_row: width * ARRAY_TEXEL,
            bytes_per_image: 0,
            size: [width, width, 1],
            texture,
            slice,
            level,
            origin: [0, 0, 0],
            plane: Plane::Whole,
        });
    }
}

/// Read every region of `texture` into its own slot of `into`.
fn read_array(work: &mut Work, texture: usize, into: usize) {
    for (at, (slice, level, width, _)) in array_regions().into_iter().enumerate() {
        work.step(Step::CopyTextureToBuffer {
            texture,
            slice,
            level,
            origin: [0, 0, 0],
            size: [width, width, 1],
            buffer: into,
            offset: slot_at(at),
            bytes_per_row: width * ARRAY_TEXEL,
            bytes_per_image: 0,
            plane: Plane::Whole,
        });
    }
}

/// A read-back buffer's bytes as a site: every byte compared, guard included, so the padding
/// around each region is a field of the answer.
fn bytes_site(name: &str, bytes: Vec<u8>) -> Node {
    let length = bytes.len() as u64;
    Node::leaf(
        name,
        Observed::bytes(bytes, ByteLayout::buffer(0, length, Element::U8)),
    )
}

/// The command buffers' statuses, in commit order.
fn statuses_site(statuses: Vec<String>) -> Node {
    Node::leaf("statuses", Observed::value(Value::Lines(statuses)))
}

/// One case of the family: its name, the program it runs, the sites it emits and what it feeds its
/// run.
struct Row {
    /// The case's name: its recording's file name, and its row in the report.
    case: &'static str,
    /// The program the case runs.
    program: fn() -> Work,
    /// The sites the case emits from what the program read.
    sites: fn(&crate::work::Done) -> Vec<Node>,
    /// What the case sets that no resource's bytes carry, as its input fingerprint.
    parameters: fn() -> Vec<u8>,
    /// The degrade notices this case's commands announce on a device that cannot take the exact
    /// form. Metal announces none, so a notice of ours is the only field of these cases that may
    /// differ at all — every byte compares exactly.
    ///
    /// **Each code here is one a venue was measured announcing for this case**, named in
    /// `NOTICES`'s own documentation with the run behind it. A code nothing was measured
    /// announcing is not listed: an unmeasured declaration is itself unchecked, and the replay
    /// reports a declaration whose notice stayed quiet as never fired.
    notices: &'static [Degradation],
}

/// A case of the family: one row.
pub struct BlitEncoderCommands {
    row: &'static Row,
}

impl Case for BlitEncoderCommands {
    fn name(&self) -> &'static str {
        self.row.case
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        self.row
            .notices
            .iter()
            .map(|kind| crate::replay::notice_allowance(kind.code()))
            .collect()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        // Every byte the program feeds its run, in resource order, and the geometry of every step:
        // either moving makes the recording's answer another program's.
        let work = (self.row.program)();
        let mut inputs = Vec::new();
        for (at, resource) in work.resources.iter().enumerate() {
            match resource {
                Resource::Buffer { bytes } => {
                    inputs.push(Input::bytes(format!("buffer:{at}"), bytes));
                }
                Resource::UnwrittenBuffer { length, storage } => {
                    let mut words = length.to_le_bytes().to_vec();
                    words.extend(storage.word().as_bytes());
                    inputs.push(Input::bytes(format!("unwritten-buffer:{at}"), &words));
                }
                Resource::Texture { spec } => {
                    inputs.push(Input::bytes(format!("texture:{at}"), &spec_words(spec)));
                }
                Resource::Fence => {
                    inputs.push(Input::bytes(format!("fence:{at}"), b"newFence"));
                }
                other => {
                    return Err(format!(
                        "{}: resource {at} is {other:?}, which this family does not state",
                        self.row.case
                    ))
                }
            }
        }
        inputs.push(Input::bytes("steps", &(self.row.parameters)()));
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let work = (self.row.program)();
        let done = made(sites, "work", side.work(&work))?;
        if done.reads.len() != work.reads() {
            return Err(Abort {
                at: "work".to_owned(),
                failure: Failure::named(
                    "UnansweredRead",
                    format!(
                        "the run answered {} of {} reads",
                        done.reads.len(),
                        work.reads()
                    ),
                ),
            });
        }
        for node in (self.row.sites)(&done) {
            sites.push(node);
        }
        Ok(())
    }
}

/// Every field of a texture a case of this family states, as the words its fingerprint covers.
fn spec_words(spec: &TextureSpec) -> Vec<u8> {
    let mut words = spec.format.to_le_bytes().to_vec();
    for axis in [
        spec.width,
        spec.height,
        spec.depth,
        spec.levels,
        spec.slices,
    ] {
        words.extend(axis.to_le_bytes());
    }
    words.extend(spec.storage.word().as_bytes());
    words.extend(spec.usage.unwrap_or(0).to_le_bytes());
    words
}

/// `copyFromTexture:toTexture:` between two arrays of the same shape.
static WHOLE_TEXTURE: Row = Row {
    case: "blit-whole-texture-copy",
    program: || {
        let mut work = Work::new();
        let source = work.add(Resource::Texture { spec: array_spec() });
        let destination = work.add(Resource::Texture { spec: array_spec() });
        let written = work.buffer(array_source(None));
        let before = work.buffer(array_source(Some(BEFORE)));
        let into = work.buffer(read_back(array_regions().len()));
        let left = work.buffer(read_back(array_regions().len()));
        // The destination's starting bytes, so every byte the case compares is one it wrote, then
        // the source's, then the one copy, then both read back: all in one command buffer, so the
        // read-back is of what this copy left rather than of a second run's.
        write_array(&mut work, before, destination);
        write_array(&mut work, written, source);
        work.step(Step::CopyWholeTexture {
            from: source,
            to: destination,
        });
        read_array(&mut work, destination, into);
        read_array(&mut work, source, left);
        work.step(Step::Commit);
        for buffer in [into, left] {
            work.step(Step::ReadBuffer {
                buffer,
                offset: 0,
                length: read_back(array_regions().len()).len(),
            });
        }
        work
    },
    sites: |done| {
        let read = |at: usize| done.reads.get(at).cloned().unwrap_or_default();
        vec![
            bytes_site("destination", read(0)),
            bytes_site("source", read(1)),
            statuses_site(done.statuses.clone()),
        ]
    },
    parameters: || b"copyFromTexture:toTexture:".to_vec(),
    notices: NOTICES,
};

/// The slices and the levels [`texture_slice_counts`] copies, from level 1 of slice 0.
const COUNTS: (u64, u64, u64, u64) = (0, 1, ARRAY_SLICES, 2);

/// `copyFromTexture:...sliceCount:levelCount:` over two slices of two levels.
static SLICE_COUNTS: Row = Row {
    case: "blit-texture-slice-and-level-counts",
    program: || {
        let (slice, level, slices, levels) = COUNTS;
        let mut work = Work::new();
        let source = work.add(Resource::Texture { spec: array_spec() });
        let destination = work.add(Resource::Texture { spec: array_spec() });
        let written = work.buffer(array_source(None));
        let before = work.buffer(array_source(Some(BEFORE)));
        let into = work.buffer(read_back(array_regions().len()));
        write_array(&mut work, before, destination);
        write_array(&mut work, written, source);
        work.step(Step::CopyTextureSlices {
            from: source,
            from_slice: slice,
            from_level: level,
            to: destination,
            to_slice: slice,
            to_level: level,
            slice_count: slices,
            level_count: levels,
        });
        read_array(&mut work, destination, into);
        work.step(Step::Commit);
        work.step(Step::ReadBuffer {
            buffer: into,
            offset: 0,
            length: read_back(array_regions().len()).len(),
        });
        work
    },
    sites: |done| {
        vec![
            bytes_site(
                "destination",
                done.reads.first().cloned().unwrap_or_default(),
            ),
            statuses_site(done.statuses.clone()),
        ]
    },
    parameters: || {
        let (slice, level, slices, levels) = COUNTS;
        [slice, level, slices, levels]
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect()
    },
    notices: NOTICES,
};

/// `copyFromTexture:...sliceCount:levelCount:` of no slices, and of no levels.
static COUNTS_OF_NONE: Row = Row {
    case: "blit-texture-counts-of-none",
    program: || {
        let mut work = Work::new();
        let source = work.add(Resource::Texture { spec: array_spec() });
        let destination = work.add(Resource::Texture { spec: array_spec() });
        let written = work.buffer(array_source(None));
        let before = work.buffer(array_source(Some(BEFORE)));
        let into = work.buffer(read_back(array_regions().len()));
        write_array(&mut work, before, destination);
        write_array(&mut work, written, source);
        for (slices, levels) in [(0, ARRAY_LEVELS), (ARRAY_SLICES, 0)] {
            work.step(Step::CopyTextureSlices {
                from: source,
                from_slice: 0,
                from_level: 0,
                to: destination,
                to_slice: 0,
                to_level: 0,
                slice_count: slices,
                level_count: levels,
            });
        }
        read_array(&mut work, destination, into);
        work.step(Step::Commit);
        work.step(Step::ReadBuffer {
            buffer: into,
            offset: 0,
            length: read_back(array_regions().len()).len(),
        });
        work
    },
    sites: |done| {
        let after = done.reads.first().cloned().unwrap_or_default();
        // The row's answer is a negative fact, so it is stated as one: a side that moved a texel
        // emits its bytes here instead, and the two differ in kind rather than in a byte nobody
        // reads twice.
        let untouched = untouched_array(&after);
        let changed = if untouched {
            Node::leaf(
                "changed",
                Observed::value(Value::Nothing(Negative::CopiedNoBytes)),
            )
        } else {
            bytes_site("changed", after.clone())
        };
        vec![
            changed,
            bytes_site("destination", after),
            statuses_site(done.statuses.clone()),
        ]
    },
    parameters: || b"sliceCount 0 levelCount 3; sliceCount 2 levelCount 0".to_vec(),
    notices: NOTICES,
};

/// Whether every region of an array read-back still holds [`BEFORE`], its guard untouched.
fn untouched_array(bytes: &[u8]) -> bool {
    if bytes.len() != read_back(array_regions().len()).len() {
        return false;
    }
    let mut want = read_back(array_regions().len());
    for (at, (_, _, _, size)) in array_regions().into_iter().enumerate() {
        let from = slot_at(at) as usize;
        want[from..from + size as usize].fill(BEFORE);
    }
    bytes == want.as_slice()
}

/// The mipmap case's texture: 8 by 8 `RGBA8Unorm`, four levels, one slice.
const MIP_WIDTH: u64 = 8;
const MIP_LEVELS: u64 = 4;
const MIP_TEXEL: u64 = 4;

/// The mipmap case's levels, as their width and the bytes each holds.
fn mip_regions() -> Vec<(u64, u64)> {
    level_widths(MIP_WIDTH, MIP_LEVELS)
        .into_iter()
        .map(|width| (width, width * width * MIP_TEXEL))
        .collect()
}

/// `generateMipmapsForTexture:` of a filterable colour texture, every level read back.
static GENERATED_MIPMAPS: Row = Row {
    case: "blit-generated-mipmaps",
    program: || {
        let mut work = Work::new();
        let spec = TextureSpec::texture_2d(PixelFormat::RGBA8Unorm.as_raw(), MIP_WIDTH, MIP_WIDTH)
            .levels(MIP_LEVELS)
            .used(TextureUsage::SHADER_READ.as_raw());
        let texture = work.add(Resource::Texture { spec });
        let level0 = (MIP_WIDTH * MIP_WIDTH * MIP_TEXEL) as usize;
        let written = work.buffer(counting(level0));
        // Every level is given a known byte first, so a level the filter did not write is a value
        // the case wrote rather than fresh memory, and only then is level 0 written with the
        // pattern the filter reads.
        let before = work.buffer(vec![BEFORE; level0]);
        let into = work.buffer(read_back(mip_regions().len()));
        for (at, (width, _)) in mip_regions().into_iter().enumerate() {
            work.step(Step::CopyBufferToTexture {
                buffer: before,
                offset: 0,
                bytes_per_row: width * MIP_TEXEL,
                bytes_per_image: 0,
                size: [width, width, 1],
                texture,
                slice: 0,
                level: at as u64,
                origin: [0, 0, 0],
                plane: Plane::Whole,
            });
        }
        work.step(Step::CopyBufferToTexture {
            buffer: written,
            offset: 0,
            bytes_per_row: MIP_WIDTH * MIP_TEXEL,
            bytes_per_image: 0,
            size: [MIP_WIDTH, MIP_WIDTH, 1],
            texture,
            slice: 0,
            level: 0,
            origin: [0, 0, 0],
            plane: Plane::Whole,
        });
        work.step(Step::GenerateMipmaps { texture });
        for (at, (width, _)) in mip_regions().into_iter().enumerate() {
            work.step(Step::CopyTextureToBuffer {
                texture,
                slice: 0,
                level: at as u64,
                origin: [0, 0, 0],
                size: [width, width, 1],
                buffer: into,
                offset: slot_at(at),
                bytes_per_row: width * MIP_TEXEL,
                bytes_per_image: 0,
                plane: Plane::Whole,
            });
        }
        work.step(Step::Commit);
        work.step(Step::ReadBuffer {
            buffer: into,
            offset: 0,
            length: read_back(mip_regions().len()).len(),
        });
        work
    },
    sites: |done| {
        vec![
            bytes_site("levels", done.reads.first().cloned().unwrap_or_default()),
            statuses_site(done.statuses.clone()),
        ]
    },
    parameters: || b"generateMipmapsForTexture: over 4 levels of 8x8 RGBA8Unorm".to_vec(),
    notices: NOTICES,
};

/// The four hints [`optimized_contents`] encodes, in that order: `None` is the whole-texture
/// selector, `Some` names a slice and a level.
const HINTS: [(bool, Option<(u64, u64)>); 4] = [
    (false, None),
    (true, Some((1, 1))),
    (false, Some((0, 0))),
    (true, None),
];

/// The four layout hints, between a copy in and a copy out of one command buffer.
static OPTIMIZED_CONTENTS: Row = Row {
    case: "blit-optimized-contents",
    program: || {
        let mut work = Work::new();
        let source = work.add(Resource::Texture { spec: array_spec() });
        let written = work.buffer(array_source(None));
        let into = work.buffer(read_back(array_regions().len()));
        write_array(&mut work, written, source);
        for (gpu, at) in HINTS {
            work.step(if gpu {
                Step::OptimizeForGpuAccess {
                    texture: source,
                    at,
                }
            } else {
                Step::OptimizeForCpuAccess {
                    texture: source,
                    at,
                }
            });
        }
        read_array(&mut work, source, into);
        work.step(Step::Commit);
        work.step(Step::ReadBuffer {
            buffer: into,
            offset: 0,
            length: read_back(array_regions().len()).len(),
        });
        work
    },
    sites: |done| {
        vec![
            bytes_site(
                "after-the-hints",
                done.reads.first().cloned().unwrap_or_default(),
            ),
            statuses_site(done.statuses.clone()),
        ]
    },
    parameters: || {
        let mut words = Vec::new();
        for (gpu, at) in HINTS {
            words.push(u8::from(gpu));
            match at {
                None => words.push(0),
                Some((slice, level)) => {
                    words.push(1);
                    words.extend(slice.to_le_bytes());
                    words.extend(level.to_le_bytes());
                }
            }
        }
        words
    },
    notices: NOTICES,
};

/// The managed buffer's length, and the fill the case is about: an unaligned range inside it, so
/// the bytes before and after it are padding the fill must not reach.
const MANAGED: u64 = 192;
const MANAGED_FILL: (u64, u64, u8) = (37, 70, 0x9a);

/// `synchronizeResource:` of a `Managed` buffer the host then reads through `contents`.
static SYNCHRONIZED_BUFFER: Row = Row {
    case: "blit-synchronized-managed-buffer",
    program: || {
        let (offset, length, value) = MANAGED_FILL;
        let mut work = Work::new();
        let managed = work.unwritten(MANAGED, Storage::Managed);
        // The whole buffer takes a known byte and is synchronized, so every byte the case reads
        // past the fill is a byte it wrote; then the fill the case is about, synchronized again.
        work.step(Step::FillBuffer {
            buffer: managed,
            offset: 0,
            length: MANAGED,
            value: GUARD,
        });
        work.step(Step::SynchronizeResource { resource: managed });
        work.step(Step::Commit);
        work.step(Step::FillBuffer {
            buffer: managed,
            offset,
            length,
            value,
        });
        work.step(Step::SynchronizeResource { resource: managed });
        work.step(Step::Commit);
        work.step(Step::ReadBuffer {
            buffer: managed,
            offset: 0,
            length: MANAGED as usize,
        });
        work
    },
    sites: |done| {
        vec![
            bytes_site("contents", done.reads.first().cloned().unwrap_or_default()),
            statuses_site(done.statuses.clone()),
        ]
    },
    parameters: || {
        let (offset, length, value) = MANAGED_FILL;
        let mut words = offset.to_le_bytes().to_vec();
        words.extend(length.to_le_bytes());
        words.push(value);
        words
    },
    notices: SPLIT_FILL,
};

/// The managed texture's extent, and the pitch the host reads it at: six times its row, so the
/// answer holds the padding between the rows and after the last one.
const MANAGED_WIDTH: u64 = 4;
const MANAGED_PITCH: u64 = MANAGED_WIDTH * ARRAY_TEXEL * 6;

/// `synchronizeTexture:slice:level:` of a `Managed` texture the host then reads with `getBytes:`.
static SYNCHRONIZED_TEXTURE: Row = Row {
    case: "blit-synchronized-managed-texture",
    program: || {
        let mut work = Work::new();
        let spec =
            TextureSpec::texture_2d(PixelFormat::R32Uint.as_raw(), MANAGED_WIDTH, MANAGED_WIDTH)
                .stored(Storage::Managed)
                .used(TextureUsage::SHADER_READ.as_raw());
        let texture = work.add(Resource::Texture { spec });
        let written = work.buffer(counting(
            (MANAGED_WIDTH * MANAGED_WIDTH * ARRAY_TEXEL) as usize,
        ));
        work.step(Step::CopyBufferToTexture {
            buffer: written,
            offset: 0,
            bytes_per_row: MANAGED_WIDTH * ARRAY_TEXEL,
            bytes_per_image: 0,
            size: [MANAGED_WIDTH, MANAGED_WIDTH, 1],
            texture,
            slice: 0,
            level: 0,
            origin: [0, 0, 0],
            plane: Plane::Whole,
        });
        work.step(Step::SynchronizeTexture {
            texture,
            slice: 0,
            level: 0,
        });
        work.step(Step::Commit);
        work.step(Step::GetBytes {
            texture,
            level: 0,
            slice: 0,
            region: Rect::new(0, 0, MANAGED_WIDTH, MANAGED_WIDTH),
            bytes_per_row: MANAGED_PITCH,
            bytes_per_image: 0,
            length: (MANAGED_PITCH * MANAGED_WIDTH) as usize,
            before: GUARD,
        });
        work
    },
    sites: |done| {
        vec![
            bytes_site("get-bytes", done.reads.first().cloned().unwrap_or_default()),
            statuses_site(done.statuses.clone()),
        ]
    },
    parameters: || {
        let mut words = MANAGED_PITCH.to_le_bytes().to_vec();
        words.push(GUARD);
        words
    },
    notices: NOTICES,
};

/// The two un-optioned selectors beside the two that take an option, over the same bytes.
static UNOPTIONED: Row = Row {
    case: "blit-unoptioned-buffer-and-texture-copies",
    program: || {
        let mut work = Work::new();
        let spec = TextureSpec::texture_2d(PixelFormat::R32Uint.as_raw(), ARRAY_WIDTH, ARRAY_WIDTH)
            .used(TextureUsage::SHADER_READ.as_raw());
        let unoptioned = work.add(Resource::Texture { spec });
        let optioned = work.add(Resource::Texture { spec });
        let row = ARRAY_WIDTH * ARRAY_TEXEL;
        let texels = (row * ARRAY_WIDTH) as usize;
        let written = work.buffer(counting(texels));
        let into = work.buffer(read_back(2));
        work.step(Step::CopyBufferToTextureWithoutOptions {
            buffer: written,
            offset: 0,
            bytes_per_row: row,
            bytes_per_image: 0,
            size: [ARRAY_WIDTH, ARRAY_WIDTH, 1],
            texture: unoptioned,
            slice: 0,
            level: 0,
            origin: [0, 0, 0],
        });
        work.step(Step::CopyBufferToTexture {
            buffer: written,
            offset: 0,
            bytes_per_row: row,
            bytes_per_image: 0,
            size: [ARRAY_WIDTH, ARRAY_WIDTH, 1],
            texture: optioned,
            slice: 0,
            level: 0,
            origin: [0, 0, 0],
            plane: Plane::Whole,
        });
        work.step(Step::CopyTextureToBufferWithoutOptions {
            texture: unoptioned,
            slice: 0,
            level: 0,
            origin: [0, 0, 0],
            size: [ARRAY_WIDTH, ARRAY_WIDTH, 1],
            buffer: into,
            offset: slot_at(0),
            bytes_per_row: row,
            bytes_per_image: 0,
        });
        work.step(Step::CopyTextureToBuffer {
            texture: optioned,
            slice: 0,
            level: 0,
            origin: [0, 0, 0],
            size: [ARRAY_WIDTH, ARRAY_WIDTH, 1],
            buffer: into,
            offset: slot_at(1),
            bytes_per_row: row,
            bytes_per_image: 0,
            plane: Plane::Whole,
        });
        work.step(Step::Commit);
        work.step(Step::ReadBuffer {
            buffer: into,
            offset: 0,
            length: read_back(2).len(),
        });
        work
    },
    sites: |done| {
        vec![
            bytes_site(
                "both-selectors",
                done.reads.first().cloned().unwrap_or_default(),
            ),
            statuses_site(done.statuses.clone()),
        ]
    },
    parameters: || b"unoptioned at slot 0; MTLBlitOptionNone at slot 1".to_vec(),
    notices: NOTICES,
};

/// The bytes [`fenced_encoders`] moves.
const FENCED: usize = 128;

/// `updateFence:` on one blit encoder of a command buffer and `waitForFence:` on the next.
static FENCED_ENCODERS: Row = Row {
    case: "blit-fenced-encoders",
    program: || {
        let mut work = Work::new();
        let fence = work.add(Resource::Fence);
        let written = work.buffer(counting(FENCED));
        let between = work.unwritten(FENCED as u64, Storage::Private);
        let into = work.buffer(vec![GUARD; FENCED]);
        // One command buffer, two encoders: the copy in, the fence update, then the wait and the
        // copy out. Committing between them would make the second encoder read what the first
        // left in another command buffer's run, which is the shape that makes a round trip pass
        // without the first half having run at all.
        work.step(Step::CopyBufferToBuffer {
            from: written,
            from_offset: 0,
            to: between,
            to_offset: 0,
            length: FENCED as u64,
        });
        work.step(Step::UpdateFence { fence });
        work.step(Step::EndEncoder);
        work.step(Step::WaitForFence { fence });
        work.step(Step::CopyBufferToBuffer {
            from: between,
            from_offset: 0,
            to: into,
            to_offset: 0,
            length: FENCED as u64,
        });
        work.step(Step::Commit);
        work.step(Step::ReadBuffer {
            buffer: into,
            offset: 0,
            length: FENCED,
        });
        work
    },
    sites: |done| {
        vec![
            bytes_site(
                "through-the-fence",
                done.reads.first().cloned().unwrap_or_default(),
            ),
            statuses_site(done.statuses.clone()),
        ]
    },
    parameters: || b"updateFence on encoder 0; waitForFence on encoder 1".to_vec(),
    notices: NOTICES,
};

/// The notices a case of this family declares, where the case provokes none of its own.
///
/// **The venue's own, and nothing else.** Each of these rows reads its answer back through a
/// `Shared` buffer, and on a device whose GPU does not share the CPU's memory such a buffer lies
/// in host-cached system memory, which the runtime announces once per device
/// (`crate::replay::shared_memory_notice` holds the whole reason). Measured on the RTX 5080 Laptop
/// (615.71.09): all eight rows that carry this list announced it and were divergent on that field
/// and nothing else (replay-nvidia.log).
///
/// Nothing of the rows' own joins it until a venue is measured announcing one. Declaring a code on
/// a row from the shape of its path would be a declaration nothing checks: a venue that does
/// announce something reads as `UndeclaredNotice`, naming the code and the case, which is the loud
/// form and the one that makes the next real notice visible.
const NOTICES: &[Degradation] = &[Degradation::SharedInSystemMemory];

/// The notice [`synchronized_managed_buffer`] announces, **measured** rather than argued.
///
/// Its second fill is `[37, 107)` of the managed buffer — neither end a multiple of four — so a
/// device whose transfer fill writes whole words splits it. Verified on an Apple M3 Max
/// (MoltenVK, API 1.2, Metal level 27.0.1): the first replay of this case reported
/// `case/notices` divergent with `metal2vulkan-degrade-split-buffer-fill` announced and nothing
/// declared, and the declaration is that run's answer. It loosens the notices site alone; every
/// byte of the buffer still compares exactly, which is the point of the case.
/// It is the one row of the family that reads its answer back out of a **`Managed`** buffer
/// rather than a `Shared` one, so [`NOTICES`]'s venue notice is not its: on the RTX 5080 Laptop
/// (615.71.09) it was the only row of the family that did not announce
/// `shared-in-system-memory`, and it compared justified there (replay-nvidia.log).
const SPLIT_FILL: &[Degradation] = &[Degradation::SplitBufferFill];

/// The case of `copyFromTexture:toTexture:`.
pub fn whole_texture_copy() -> BlitEncoderCommands {
    BlitEncoderCommands {
        row: &WHOLE_TEXTURE,
    }
}

/// The case of the slice-and-level-count copy.
pub fn texture_slice_counts() -> BlitEncoderCommands {
    BlitEncoderCommands { row: &SLICE_COUNTS }
}

/// The case of a copy of no slices and of no levels.
pub fn texture_counts_of_none() -> BlitEncoderCommands {
    BlitEncoderCommands {
        row: &COUNTS_OF_NONE,
    }
}

/// The case of `generateMipmapsForTexture:`.
pub fn generated_mipmaps() -> BlitEncoderCommands {
    BlitEncoderCommands {
        row: &GENERATED_MIPMAPS,
    }
}

/// The case of the four layout hints.
pub fn optimized_contents() -> BlitEncoderCommands {
    BlitEncoderCommands {
        row: &OPTIMIZED_CONTENTS,
    }
}

/// The case of `synchronizeResource:`.
pub fn synchronized_managed_buffer() -> BlitEncoderCommands {
    BlitEncoderCommands {
        row: &SYNCHRONIZED_BUFFER,
    }
}

/// The case of `synchronizeTexture:slice:level:`.
pub fn synchronized_managed_texture() -> BlitEncoderCommands {
    BlitEncoderCommands {
        row: &SYNCHRONIZED_TEXTURE,
    }
}

/// The case of the two un-optioned copy selectors.
pub fn unoptioned_copies() -> BlitEncoderCommands {
    BlitEncoderCommands { row: &UNOPTIONED }
}

/// The case of a fence between two of a command buffer's encoders.
pub fn fenced_encoders() -> BlitEncoderCommands {
    BlitEncoderCommands {
        row: &FENCED_ENCODERS,
    }
}

/// Every case of the family, in the order the report lists them.
pub fn all() -> Vec<Box<dyn Case>> {
    vec![
        Box::new(whole_texture_copy()),
        Box::new(texture_slice_counts()),
        Box::new(texture_counts_of_none()),
        Box::new(generated_mipmaps()),
        Box::new(optimized_contents()),
        Box::new(synchronized_managed_buffer()),
        Box::new(synchronized_managed_texture()),
        Box::new(unoptioned_copies()),
        Box::new(fenced_encoders()),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// Every row of the family, for the guards that read each one.
    const EVERY_ROW: [&Row; 9] = [
        &WHOLE_TEXTURE,
        &SLICE_COUNTS,
        &COUNTS_OF_NONE,
        &GENERATED_MIPMAPS,
        &OPTIMIZED_CONTENTS,
        &SYNCHRONIZED_BUFFER,
        &SYNCHRONIZED_TEXTURE,
        &UNOPTIONED,
        &FENCED_ENCODERS,
    ];

    /// The cases name distinct recordings, and `all` holds every row.
    #[test]
    fn the_cases_are_distinct_and_complete() {
        let named: BTreeSet<&str> = all().iter().map(|case| case.name()).collect();
        assert_eq!(named.len(), all().len());
        assert_eq!(named.len(), EVERY_ROW.len());
        let rows: BTreeSet<&str> = EVERY_ROW.iter().map(|row| row.case).collect();
        assert_eq!(rows, named);
    }

    /// Every case states its inputs, and every program answers at least one read, so no case can
    /// record an answer nothing read.
    #[test]
    fn every_case_states_its_inputs_and_reads_something() {
        for case in all() {
            let inputs = case
                .inputs()
                .unwrap_or_else(|why| panic!("{}: {why}", case.name()));
            assert!(!inputs.is_empty(), "{}", case.name());
        }
        for row in EVERY_ROW {
            assert!((row.program)().reads() > 0, "{}", row.case);
        }
    }

    /// Every case loosens the notices site and nothing else: no byte of these cases is ever
    /// loosened, whatever form a device's copy or fill takes.
    ///
    /// And the declaration is held to the one case it was measured on. A code named on a case no
    /// venue was measured announcing it for is a declaration nothing checks, and the replay would
    /// report it as an unfired allowance rather than failing — which is the quiet direction, so it
    /// is pinned here instead.
    #[test]
    fn only_the_measured_case_declares_a_notice() {
        for case in all() {
            for allowance in case.allowances() {
                assert_eq!(
                    allowance.path,
                    format!("case/{}/*", crate::replay::NOTICES_SITE),
                    "{}",
                    case.name()
                );
            }
        }
        // Every row declares the venue's memory notice except the one that reads its answer back
        // out of a `Managed` buffer, which declares its own measured notice instead.
        assert_eq!(NOTICES, [Degradation::SharedInSystemMemory]);
        let declaring: Vec<&str> = EVERY_ROW
            .iter()
            .filter(|row| row.notices != NOTICES)
            .map(|row| row.case)
            .collect();
        assert_eq!(declaring, [SYNCHRONIZED_BUFFER.case]);
        assert_eq!(SYNCHRONIZED_BUFFER.notices, SPLIT_FILL);
    }

    /// The fill the declaring case makes really is the shape the notice is about: neither end of
    /// its range is a multiple of four, so a device whose fill writes whole words has to split it.
    ///
    /// Without this the declaration would rest on a sentence in a doc comment, and a later edit
    /// that aligned the range would leave a code declared that nothing provokes.
    #[test]
    fn the_declaring_cases_fill_is_unaligned_at_both_ends() {
        let (offset, length, _) = MANAGED_FILL;
        assert_ne!(offset % 4, 0, "the fill starts on a word boundary");
        assert_ne!((offset + length) % 4, 0, "the fill ends on a word boundary");
        assert!(offset + length <= MANAGED);
    }

    /// Assert that `origin` plus `size` lies inside `extent` on every axis, naming the step.
    fn inside(case: &str, step: usize, origin: &[u64; 3], size: &[u64; 3], extent: &[u64; 3]) {
        for (axis, ((from, span), bound)) in origin
            .iter()
            .zip(size.iter())
            .zip(extent.iter())
            .enumerate()
        {
            assert!(
                from + span <= *bound,
                "{case}: step {step} reaches {} of {bound} on axis {axis}",
                from + span
            );
        }
    }

    /// Every texture region every program names lies inside its texture's level and slice, and
    /// every buffer range inside its buffer, so no program can hand a device an out-of-bounds
    /// transfer.
    ///
    /// The counts of none are the exception the row is about: a count of 0 names slices and levels
    /// that exist and copies nothing.
    #[test]
    fn every_region_lies_inside_its_texture() {
        for row in EVERY_ROW {
            let work = (row.program)();
            let spec = |at: usize| match &work.resources[at] {
                Resource::Texture { spec } => *spec,
                other => panic!("{}: resource {at} is {other:?}, not a texture", row.case),
            };
            let buffer_length = |at: usize| match &work.resources[at] {
                Resource::Buffer { bytes } => bytes.len() as u64,
                Resource::UnwrittenBuffer { length, .. } => *length,
                other => panic!("{}: resource {at} is {other:?}, not a buffer", row.case),
            };
            // A level's extent, which every region of it must fit inside.
            let level_extent = |at: usize, level: u64| {
                let spec = spec(at);
                assert!(level < spec.levels, "{}: level {level}", row.case);
                [
                    (spec.width >> level).max(1),
                    (spec.height >> level).max(1),
                    1,
                ]
            };
            for (at, step) in work.steps.iter().enumerate() {
                match step {
                    Step::CopyBufferToTexture {
                        buffer,
                        offset,
                        bytes_per_row,
                        size,
                        texture,
                        slice,
                        level,
                        origin,
                        ..
                    }
                    | Step::CopyBufferToTextureWithoutOptions {
                        buffer,
                        offset,
                        bytes_per_row,
                        size,
                        texture,
                        slice,
                        level,
                        origin,
                        ..
                    } => {
                        let extent = level_extent(*texture, *level);
                        assert!(*slice < spec(*texture).faces(), "{}: step {at}", row.case);
                        inside(row.case, at, origin, size, &extent);
                        assert!(
                            offset + bytes_per_row * size[1] <= buffer_length(*buffer),
                            "{}: step {at} reads past its buffer",
                            row.case
                        );
                    }
                    Step::CopyTextureToBuffer {
                        texture,
                        slice,
                        level,
                        origin,
                        size,
                        buffer,
                        offset,
                        bytes_per_row,
                        ..
                    }
                    | Step::CopyTextureToBufferWithoutOptions {
                        texture,
                        slice,
                        level,
                        origin,
                        size,
                        buffer,
                        offset,
                        bytes_per_row,
                        ..
                    } => {
                        let extent = level_extent(*texture, *level);
                        assert!(*slice < spec(*texture).faces(), "{}: step {at}", row.case);
                        inside(row.case, at, origin, size, &extent);
                        assert!(
                            offset + bytes_per_row * size[1] <= buffer_length(*buffer),
                            "{}: step {at} writes past its buffer",
                            row.case
                        );
                    }
                    Step::CopyTextureSlices {
                        from,
                        from_slice,
                        from_level,
                        to,
                        to_slice,
                        to_level,
                        slice_count,
                        level_count,
                    } => {
                        if *slice_count == 0 || *level_count == 0 {
                            continue;
                        }
                        for (at_texture, slice, level) in [
                            (*from, *from_slice, *from_level),
                            (*to, *to_slice, *to_level),
                        ] {
                            let spec = spec(at_texture);
                            assert!(
                                slice + slice_count <= spec.faces()
                                    && level + level_count <= spec.levels,
                                "{}: step {at} runs past its texture",
                                row.case
                            );
                        }
                    }
                    Step::SynchronizeTexture {
                        texture,
                        slice,
                        level,
                    } => {
                        let spec = spec(*texture);
                        assert!(
                            *slice < spec.faces() && *level < spec.levels,
                            "{}: step {at} names a slice or level past its texture",
                            row.case
                        );
                    }
                    Step::OptimizeForGpuAccess { texture, at: named }
                    | Step::OptimizeForCpuAccess { texture, at: named } => {
                        if let Some((slice, level)) = named {
                            let spec = spec(*texture);
                            assert!(
                                *slice < spec.faces() && *level < spec.levels,
                                "{}: step {at} names a slice or level past its texture",
                                row.case
                            );
                        }
                    }
                    Step::GetBytes {
                        texture,
                        level,
                        slice,
                        region,
                        bytes_per_row,
                        length,
                        ..
                    } => {
                        let extent = level_extent(*texture, *level);
                        assert!(*slice < spec(*texture).faces(), "{}: step {at}", row.case);
                        inside(row.case, at, &region.origin, &region.size, &extent);
                        assert!(
                            bytes_per_row * region.size[1] <= *length as u64,
                            "{}: step {at} reads into fewer bytes than its pitch needs",
                            row.case
                        );
                    }
                    Step::FillBuffer {
                        buffer,
                        offset,
                        length,
                        ..
                    } => assert!(
                        offset + length <= buffer_length(*buffer),
                        "{}: step {at} fills past its buffer",
                        row.case
                    ),
                    Step::CopyBufferToBuffer {
                        from,
                        from_offset,
                        to,
                        to_offset,
                        length,
                    } => {
                        assert!(
                            from_offset + length <= buffer_length(*from)
                                && to_offset + length <= buffer_length(*to),
                            "{}: step {at} copies past a buffer",
                            row.case
                        );
                    }
                    Step::ReadBuffer {
                        buffer,
                        offset,
                        length,
                    } => assert!(
                        offset + *length as u64 <= buffer_length(*buffer),
                        "{}: step {at} reads past its buffer",
                        row.case
                    ),
                    _ => {}
                }
            }
        }
    }

    /// A read-back slot is wider than the widest region any case copies into one, so each slot
    /// really holds padding after its region rather than running into the next.
    ///
    /// Without this the guard bytes would be a decoration: a slot exactly as wide as its region
    /// compares the same bytes twice and says nothing about a copy that ran long.
    #[test]
    fn every_slot_is_wider_than_the_region_it_holds() {
        let widest = array_regions()
            .into_iter()
            .map(|(_, _, _, bytes)| bytes)
            .chain(mip_regions().into_iter().map(|(_, bytes)| bytes))
            .max()
            .expect("the families name regions");
        assert!(
            widest < SLOT as u64,
            "a region of {widest} bytes does not fit a slot of {SLOT} with padding left over"
        );
        assert_eq!(
            read_back(1).len(),
            LEAD + SLOT,
            "the read-back's lead-in is what gives the first slot its guard before the region"
        );
    }

    /// The array source's offsets tile its buffer exactly: each region starts where the one before
    /// it ended, and the last ends at the buffer's length.
    ///
    /// This is what makes the copies' offsets a statement rather than an arithmetic accident; it
    /// fails if a level's size or the region order ever moves.
    #[test]
    fn the_array_source_tiles_its_buffer() {
        let regions = array_regions();
        let mut at = 0u64;
        for (index, (_, _, _, bytes)) in regions.iter().enumerate() {
            assert_eq!(array_offset(index), at, "region {index}");
            at += bytes;
        }
        assert_eq!(at as usize, array_source(None).len());
        assert_eq!(
            regions.len() as u64,
            ARRAY_SLICES * ARRAY_LEVELS,
            "every slice names every level"
        );
    }

    /// The bytes a committed recording holds at `site`.
    fn recorded(case: &dyn Case, site: &str) -> Vec<u8> {
        let recording = crate::recording::read(case.name())
            .unwrap_or_else(|why| panic!("{}: {why}", case.name()));
        let answer = recording
            .answers
            .first()
            .expect("the recording holds an answer");
        let outcome = answer
            .outcomes
            .first()
            .expect("the answer holds an outcome");
        let node = outcome
            .transcript
            .root
            .child(site)
            .unwrap_or_else(|| panic!("{}: the recording holds no {site}", case.name()));
        match &node.observed.value {
            Value::Bytes(bytes) => bytes.clone(),
            other => panic!("{site}: the recording holds {}", other.kind()),
        }
    }

    /// A read-back of the array textures in which region `at` holds `bytes[at]`, guard elsewhere:
    /// what a case's own statement of the copy says Metal must have produced.
    fn expected_array(per_region: impl Fn(usize) -> Vec<u8>) -> Vec<u8> {
        let mut want = read_back(array_regions().len());
        for (at, (_, _, _, size)) in array_regions().into_iter().enumerate() {
            let from = slot_at(at) as usize;
            let bytes = per_region(at);
            assert_eq!(bytes.len(), size as usize, "region {at}");
            want[from..from + size as usize].copy_from_slice(&bytes);
        }
        want
    }

    /// The region `at` of an [`array_source`] buffer.
    fn source_region(at: usize) -> Vec<u8> {
        let (_, _, _, size) = array_regions()[at];
        let from = array_offset(at) as usize;
        array_source(None)[from..from + size as usize].to_vec()
    }

    /// **Metal pairs every level and every slice** for `copyFromTexture:toTexture:` between two
    /// textures of one shape, and leaves the source as it was — and the guard around each region
    /// of the read-back is untouched.
    ///
    /// This is the cross-check the case exists for: the selector names no region, so what it
    /// copies is Metal's own rule and nothing in this tree stated it before.
    #[test]
    fn metals_whole_texture_copy_moved_every_level_and_slice() {
        let case = whole_texture_copy();
        assert_eq!(
            recorded(&case, "destination"),
            expected_array(source_region)
        );
        assert_eq!(recorded(&case, "source"), expected_array(source_region));
    }

    /// The slice-and-level-count copy moved **only** the levels and slices its counts name: level
    /// 0 of both slices still reads what the case wrote into it.
    #[test]
    fn metals_slice_counts_moved_the_levels_the_counts_name() {
        let (_, from_level, _, levels) = COUNTS;
        let moved = |at: usize| {
            let (_, level, _, size) = array_regions()[at];
            if level >= from_level && level < from_level + levels {
                source_region(at)
            } else {
                vec![BEFORE; size as usize]
            }
        };
        assert_eq!(
            recorded(&texture_slice_counts(), "destination"),
            expected_array(moved)
        );
        // And the case really does leave a level out, or the assertion above would be the
        // whole-texture copy's under another name.
        assert!(
            array_regions()
                .iter()
                .any(|(_, level, _, _)| *level < from_level || *level >= from_level + levels),
            "the counts cover every level, so the row says nothing the whole-texture copy does not"
        );
    }

    /// The four layout hints changed no byte: every region reads what the case copied in, inside
    /// the same command buffer the hints were encoded in.
    #[test]
    fn metals_layout_hints_changed_no_byte() {
        assert_eq!(
            recorded(&optimized_contents(), "after-the-hints"),
            expected_array(source_region)
        );
    }

    /// `synchronizeResource:` made the GPU's fill visible to the host, and the bytes outside the
    /// fill are the ones the case wrote before it.
    #[test]
    fn metals_synchronized_buffer_holds_the_fill_and_its_padding() {
        let (offset, length, value) = MANAGED_FILL;
        let mut want = vec![GUARD; MANAGED as usize];
        want[offset as usize..(offset + length) as usize].fill(value);
        assert_eq!(recorded(&synchronized_managed_buffer(), "contents"), want);
    }

    /// `synchronizeTexture:slice:level:` made the texels visible to `getBytes:`, each row at the
    /// pitch the case read at — and **the bytes between the rows and after the last one stayed
    /// the guard**, which is the half a read into zeroed memory could not answer.
    #[test]
    fn metals_synchronized_texture_left_the_padding_alone() {
        let row = (MANAGED_WIDTH * ARRAY_TEXEL) as usize;
        let pitch = MANAGED_PITCH as usize;
        let texels = counting(row * MANAGED_WIDTH as usize);
        let mut want = vec![GUARD; pitch * MANAGED_WIDTH as usize];
        for at in 0..MANAGED_WIDTH as usize {
            want[at * pitch..at * pitch + row].copy_from_slice(&texels[at * row..(at + 1) * row]);
        }
        assert_eq!(recorded(&synchronized_managed_texture(), "get-bytes"), want);
        assert!(pitch > row, "the read pitch is no wider than a row");
    }

    /// Metal's two copy selectors — the one that takes an `MTLBlitOption` and the one that does
    /// not — moved the same bytes into their own slots, each slot's guard intact.
    ///
    /// This is the whole claim the un-optioned case can make, and it is Metal's to keep: our own
    /// leg reaches one method for both selectors, so a difference here would be Metal's.
    #[test]
    fn metals_two_copy_selectors_moved_the_same_bytes() {
        let bytes = recorded(&unoptioned_copies(), "both-selectors");
        let row = (ARRAY_WIDTH * ARRAY_TEXEL) as usize;
        let texels = row * ARRAY_WIDTH as usize;
        let mut want = read_back(2);
        for slot in 0..2 {
            let from = slot_at(slot) as usize;
            want[from..from + texels].copy_from_slice(&counting(texels));
        }
        assert_eq!(bytes, want);
    }

    /// `generateMipmapsForTexture:` wrote every level under level 0 — none of them still reads the
    /// byte the case put there — and left level 0 exactly as the case wrote it.
    ///
    /// The filtered values themselves are Metal's, held by the recording rather than restated
    /// here: an arithmetic of our own beside them would be a second implementation to keep in
    /// step, and the recording is the oracle.
    #[test]
    fn metals_mipmaps_wrote_every_level_under_the_top() {
        let bytes = recorded(&generated_mipmaps(), "levels");
        let regions = mip_regions();
        let level0 = regions[0].1 as usize;
        assert_eq!(
            &bytes[LEAD..LEAD + level0],
            counting(level0).as_slice(),
            "the filter moved level 0"
        );
        for (at, (_, size)) in regions.iter().enumerate().skip(1) {
            let from = slot_at(at) as usize;
            let level = &bytes[from..from + *size as usize];
            assert!(
                level.iter().any(|byte| *byte != BEFORE),
                "level {at} still reads the byte the case wrote, so the filter wrote nothing"
            );
            assert_eq!(
                &bytes[from + *size as usize..from + SLOT],
                vec![GUARD; SLOT - *size as usize].as_slice(),
                "level {at}'s guard moved"
            );
        }
    }

    /// The untouched-array fact is a fact: it holds of a read-back in which every region reads
    /// [`BEFORE`] and the guard is intact, and fails when one byte of either moves.
    ///
    /// Stated both ways because a predicate only ever checked on the passing side is a predicate
    /// that could be `true`.
    #[test]
    fn the_untouched_fact_fails_on_a_byte_that_moved() {
        let mut bytes = read_back(array_regions().len());
        for (at, (_, _, _, size)) in array_regions().into_iter().enumerate() {
            let from = slot_at(at) as usize;
            bytes[from..from + size as usize].fill(BEFORE);
        }
        assert!(untouched_array(&bytes));
        for moved in [0usize, LEAD, LEAD + 1, bytes.len() - 1] {
            let mut other = bytes.clone();
            other[moved] ^= 0xff;
            assert!(
                !untouched_array(&other),
                "a byte moved at {moved} and the fact still held"
            );
        }
        assert!(
            !untouched_array(&bytes[..bytes.len() - 1]),
            "a short read still held the fact"
        );
    }
}
