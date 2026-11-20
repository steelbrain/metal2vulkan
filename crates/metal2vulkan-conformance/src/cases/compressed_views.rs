//! How block-compressed blocks move through a format view and into interior regions.
//!
//! The first lane moves whole levels; this one asks the three questions that lane adds, in the same
//! five shapes ([`crate::cases::compressed_moves::SHAPES`]) and over the same fourteen formats:
//!
//! - `viewout`: every level and slice put whole by `replaceRegion:`, then blitted out tight
//!   **through a view in the sibling format** — the sRGB twin, or the other member of a signed or
//!   float pair — made **before** the texture held anything, so the view cannot have taken a copy
//!   of what was there;
//! - `insub`: a buffer blitted into interior regions of levels 0 and 1 of the **last** slice, from
//!   texel (4, 4) to the level's edges, then the texture out tight;
//! - `viewin`: one block blitted from a buffer **through the view**, then the texture out tight;
//! - `putsub`: `replaceRegion:` of an interior region of level 0, slice 0, from texel (4, 4) to its
//!   edges, then the texture out tight.
//!
//! **Metal's own bytes.** Record mode runs the steps on an Apple M3 Max and each recording holds
//! Metal's own bytes. A byte model in Rust stays as a cross-check (the test
//! `every_recorded_view_move_is_the_byte_model`); its other tie to Metal is one SHA-256 over 336
//! lines of six-byte digests an Apple **M4** printed.
//!
//! **A view is a resource, not a step**, so the suite's interpreters make it before the first step
//! runs, so the view is made before the texture holds anything, with no special vocabulary. A
//! signed or float pair's view needs `MTLTextureUsagePixelFormatView`, which the sRGB
//! twins do not ([`sibling`]), and the case states the usage bits rather than letting either side
//! infer them.
//!
//! **One case per format**, as the rest of the compressed families: a venue with no texture of one
//! answers its typed refusal there and counts as lacking that format.
//!
//! **Not a case:** a 1030 x 520 texture of all eleven levels, whose level 0 alone is over 500 KiB
//! of blocks, so a whole-value recording of its four steps would be several MiB. It wants the
//! recording's digest-plus-window form, which no case has exercised yet.
//!
//! Nothing here can fault a GPU: host copies and blit copies between textures the program makes, no
//! dispatch, no draw, no loop, and every region block-aligned and inside its level.

use crate::allowance::Allowance;
use crate::case::{made, Abort, Case, Failure, Input, MetalSource, Side};
use crate::cases::compressed_moves::{out_tight, Expected, Model, Rng, Shape, SHAPES};
use crate::transcript::{Node, Observed, Sites, Value};
use crate::work::{Plane, Rect, Resource, Step, Work};
use metal2vulkan::metal::{PixelFormat, TextureUsage};

/// One BC format: the case's name, its raw `MTLPixelFormat` (stated as
/// `PixelFormat::<name>.as_raw()`, so the value is named in the case's own text and not only
/// in a comment beside it), its block's bytes, and the sibling
/// format its view takes.
struct Format {
    name: &'static str,
    format: u32,
    block: u64,
    /// The sibling format and whether a view of it needs `MTLTextureUsagePixelFormatView`.
    sibling: (u32, bool),
}

/// Every BC format, in a fixed order, because a shape's seed is its place in that order.
///
/// The siblings: an sRGB twin either way (no `PixelFormatView` usage, since Metal
/// counts the twins as one layout), and the other member of a signed or a float pair (which needs
/// it).
const FORMATS: [Format; 14] = [
    Format {
        name: "compressed-views-bc1-rgba",
        format: PixelFormat::BC1RGBA.as_raw(),
        block: 8,
        sibling: (PixelFormat::BC1RGBASRGB.as_raw(), false),
    },
    Format {
        name: "compressed-views-bc1-rgba-srgb",
        format: PixelFormat::BC1RGBASRGB.as_raw(),
        block: 8,
        sibling: (PixelFormat::BC1RGBA.as_raw(), false),
    },
    Format {
        name: "compressed-views-bc2-rgba",
        format: PixelFormat::BC2RGBA.as_raw(),
        block: 16,
        sibling: (PixelFormat::BC2RGBASRGB.as_raw(), false),
    },
    Format {
        name: "compressed-views-bc2-rgba-srgb",
        format: PixelFormat::BC2RGBASRGB.as_raw(),
        block: 16,
        sibling: (PixelFormat::BC2RGBA.as_raw(), false),
    },
    Format {
        name: "compressed-views-bc3-rgba",
        format: PixelFormat::BC3RGBA.as_raw(),
        block: 16,
        sibling: (PixelFormat::BC3RGBASRGB.as_raw(), false),
    },
    Format {
        name: "compressed-views-bc3-rgba-srgb",
        format: PixelFormat::BC3RGBASRGB.as_raw(),
        block: 16,
        sibling: (PixelFormat::BC3RGBA.as_raw(), false),
    },
    Format {
        name: "compressed-views-bc4-r-unorm",
        format: PixelFormat::BC4RUnorm.as_raw(),
        block: 8,
        sibling: (PixelFormat::BC4RSnorm.as_raw(), true),
    },
    Format {
        name: "compressed-views-bc4-r-snorm",
        format: PixelFormat::BC4RSnorm.as_raw(),
        block: 8,
        sibling: (PixelFormat::BC4RUnorm.as_raw(), true),
    },
    Format {
        name: "compressed-views-bc5-rg-unorm",
        format: PixelFormat::BC5RGUnorm.as_raw(),
        block: 16,
        sibling: (PixelFormat::BC5RGSnorm.as_raw(), true),
    },
    Format {
        name: "compressed-views-bc5-rg-snorm",
        format: PixelFormat::BC5RGSnorm.as_raw(),
        block: 16,
        sibling: (PixelFormat::BC5RGUnorm.as_raw(), true),
    },
    Format {
        name: "compressed-views-bc6h-rgb-float",
        format: PixelFormat::BC6HRGBFloat.as_raw(),
        block: 16,
        sibling: (PixelFormat::BC6HRGBUfloat.as_raw(), true),
    },
    Format {
        name: "compressed-views-bc6h-rgb-ufloat",
        format: PixelFormat::BC6HRGBUfloat.as_raw(),
        block: 16,
        sibling: (PixelFormat::BC6HRGBFloat.as_raw(), true),
    },
    Format {
        name: "compressed-views-bc7-rgba-unorm",
        format: PixelFormat::BC7RGBAUnorm.as_raw(),
        block: 16,
        sibling: (PixelFormat::BC7RGBAUnormSRGB.as_raw(), false),
    },
    Format {
        name: "compressed-views-bc7-rgba-unorm-srgb",
        format: PixelFormat::BC7RGBAUnormSRGB.as_raw(),
        block: 16,
        sibling: (PixelFormat::BC7RGBAUnorm.as_raw(), false),
    },
];

/// The sibling format of the raw `MTLPixelFormat` `raw`, and whether viewing it needs
/// `MTLTextureUsagePixelFormatView`.
pub fn sibling(raw: u32) -> Option<(u32, bool)> {
    FORMATS
        .iter()
        .find(|format| format.format == raw)
        .map(|format| format.sibling)
}

/// The region from texel (4, 4) — and depth 1 of a 3D texture deeper than one image — of level `l`
/// to its edges, as an origin and a size in texels and its first block; `None` where the level is a
/// block or less across or down.
fn interior(shape: &Shape, l: u64) -> Option<([u64; 3], [u64; 3], [u64; 3])> {
    let [w, h, d] = shape.level(l);
    if w <= 4 || h <= 4 {
        return None;
    }
    let z = u64::from(shape.three_d() && d > 1);
    Some(([4, 4, z], [w - 4, h - 4, d - z], [1, 1, z]))
}

/// A shape's program and the model's bytes for each of its reads, in the order it asks for them.
pub struct Plan {
    pub work: Work,
    pub expected: Vec<Expected>,
}

/// The plan for `format` in `shape`, seeded by its place.
///
/// The one function builds the program and the expectation together, so their offsets cannot drift
/// apart.
fn plan(format: &Format, shape: &Shape, seed: u64) -> Plan {
    let block = format.block;
    let mut rng = Rng(seed);
    let mut work = Work::new();
    let mut expected = Vec::new();
    let three_d = shape.three_d();
    let (view_format, needs_usage) = format.sibling;
    // A signed or float pair's view needs `PixelFormatView`; the sRGB twins do not, and the case
    // separates the two because a texture that may be viewed in another format is laid out
    // differently from one that may not.
    let usage = if needs_usage {
        (TextureUsage::SHADER_READ | TextureUsage::PIXEL_FORMAT_VIEW).as_raw()
    } else {
        TextureUsage::SHADER_READ.as_raw()
    };
    let first = work.add(Resource::Texture {
        spec: shape.spec(format.format, false).used(usage),
    });
    // Made before any step runs, before the texture holds anything: the
    // interpreters make every resource first, so the view cannot have taken a copy of what was
    // there.
    let view = work.add(Resource::View {
        texture: first,
        format: view_format,
        level: 0,
        level_count: shape.levels,
        slice: 0,
        slice_count: shape.slices(),
        flatten: false,
    });
    let mut model = Model::default();

    // viewout: every level and slice put whole, rows two blocks past, then out tight through the
    // view.
    for (l, s) in shape.each() {
        let [w, h, d] = shape.level(l);
        let [bx, by, bz] = shape.blocks(l);
        let row = bx * block + 2 * block;
        let image = if three_d { row * by + block } else { 0 };
        let total = if three_d { image * bz } else { row * by };
        let bytes = rng.bytes(total);
        model.take((l, s), [0; 3], [bx, by, bz], (&bytes, 0, row, image), block);
        work.step(Step::ReplaceRegion {
            texture: first,
            level: l,
            slice: s,
            region: Rect::new_3d([0, 0, 0], [w, h, d]),
            bytes,
            rows: by,
            bytes_per_row: row,
            bytes_per_image: image,
        });
    }
    out_tight(
        &mut work,
        &mut expected,
        shape,
        block,
        view,
        &model,
        "viewout",
    );

    // insub: levels 0 and 1 of the last slice, from texel (4, 4) to the edges.
    let last = shape.slices() - 1;
    let mut places = Vec::new();
    let mut at = 0;
    for l in 0..shape.levels.min(2) {
        let Some((origin, size, first_block)) = interior(shape, l) else {
            continue;
        };
        let [bx, by, _] = shape.blocks(l);
        let count = [bx - first_block[0], by - first_block[1], size[2]];
        let row = count[0] * block + block;
        let image = shape.image(l, row, block);
        at += 2 * block;
        places.push((l, origin, size, first_block, count, at, row, image));
        at += if three_d {
            image * count[2]
        } else {
            row * count[1]
        };
    }
    let source = rng.bytes(at.max(1));
    let source_buffer = work.buffer(source.clone());
    for &(l, origin, size, _, _, offset, row, image) in &places {
        work.step(Step::CopyBufferToTexture {
            buffer: source_buffer,
            offset,
            bytes_per_row: row,
            bytes_per_image: image,
            size,
            texture: first,
            slice: last,
            level: l,
            origin,
            plane: Plane::Whole,
        });
    }
    work.step(Step::Commit);
    for &(l, _, _, first_block, count, offset, row, image) in &places {
        let image = if three_d { image } else { 0 };
        model.take(
            (l, last),
            first_block,
            count,
            (&source, offset, row, image),
            block,
        );
    }
    out_tight(
        &mut work,
        &mut expected,
        shape,
        block,
        first,
        &model,
        "insub",
    );

    // viewin: one block blitted through the view.
    let [w, h, _] = shape.level(0);
    let one = rng.bytes(block);
    let one_buffer = work.buffer(one.clone());
    work.step(Step::CopyBufferToTexture {
        buffer: one_buffer,
        offset: 0,
        bytes_per_row: block,
        bytes_per_image: if three_d { block } else { 0 },
        size: [w.min(4), h.min(4), 1],
        texture: view,
        slice: 0,
        level: 0,
        origin: [0, 0, 0],
        plane: Plane::Whole,
    });
    work.step(Step::Commit);
    model.take((0, 0), [0; 3], [1, 1, 1], (&one, 0, block, 0), block);
    out_tight(
        &mut work,
        &mut expected,
        shape,
        block,
        first,
        &model,
        "viewin",
    );

    // putsub: replaceRegion of level 0, slice 0, from texel (4, 4) to the edges, rows one block
    // past.
    if let Some((origin, size, first_block)) = interior(shape, 0) {
        let [bx, by, _] = shape.blocks(0);
        let count = [bx - first_block[0], by - first_block[1], size[2]];
        let row = count[0] * block + block;
        let image = if three_d { row * count[1] + block } else { 0 };
        let total = if three_d {
            image * count[2]
        } else {
            row * count[1]
        };
        let bytes = rng.bytes(total);
        model.take((0, 0), first_block, count, (&bytes, 0, row, image), block);
        work.step(Step::ReplaceRegion {
            texture: first,
            level: 0,
            slice: 0,
            region: Rect::new_3d(origin, size),
            bytes,
            rows: count[1],
            bytes_per_row: row,
            bytes_per_image: image,
        });
    }
    out_tight(
        &mut work,
        &mut expected,
        shape,
        block,
        first,
        &model,
        "putsub",
    );
    Plan { work, expected }
}

/// The case for one format.
pub struct CompressedViews {
    at: usize,
}

/// Every format's case, in [`FORMATS`] order.
pub fn all() -> Vec<CompressedViews> {
    (0..FORMATS.len())
        .map(|at| CompressedViews { at })
        .collect()
}

impl CompressedViews {
    /// The format the case moves.
    fn format(&self) -> &'static Format {
        &FORMATS[self.at]
    }

    /// The plans of every shape, seeded by the second seed base, the block size and the shape's
    /// place.
    pub fn plans(&self) -> Vec<(&'static Shape, Plan)> {
        let format = self.format();
        SHAPES
            .iter()
            .enumerate()
            .map(|(k, shape)| {
                let seed = 0x5eed_1000 + format.block * 0x100 + k as u64;
                (shape, plan(format, shape, seed))
            })
            .collect()
    }
}

impl Case for CompressedViews {
    fn name(&self) -> &'static str {
        FORMATS[self.at].name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // The same notices the first lane's cases name, for the same reason and from the same
        // measurement: one mechanism, not two. A view of a BC texture reads the device's decode
        // of its blocks, which the runtime announces as `compressed-texel-decode` where it makes
        // it.
        crate::cases::compressed_moves::block_allowances()
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let format = self.format();
        let mut inputs = vec![Input::bytes(
            "formats",
            &[format.format, format.sibling.0, u32::from(format.sibling.1)]
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<u8>>(),
        )];
        // The whole program of every shape, as the interpreter will run it: every descriptor, the
        // view's ranges, every region, every pitch and every byte written.
        for (shape, plan) in self.plans() {
            inputs.push(Input::bytes(
                format!("program:{}", shape.name),
                &crate::cases::compressed_moves::fingerprint(&plan.work),
            ));
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        for (shape, plan) in self.plans() {
            let done = made(
                sites,
                &format!("{}/work", shape.name),
                side.work(&plan.work),
            )?;
            let mut nodes: Vec<Node> = Vec::new();
            for (at, expected) in plan.expected.iter().enumerate() {
                let bytes = done.read(at).map_err(|why| Abort {
                    at: format!("{}/{}/{}", shape.name, expected.step, expected.leaf),
                    failure: Failure::named("UnansweredRead", why),
                })?;
                nodes.push(Node::branch(
                    expected.step.to_owned(),
                    vec![Node::leaf(
                        expected.leaf.clone(),
                        Observed::bytes(bytes.to_vec(), expected.layout.clone()),
                    )],
                ));
            }
            sites.push(Node::branch(shape.name.to_owned(), nodes));
            sites.value(
                format!("{}/statuses", shape.name),
                Value::Lines(done.statuses.clone()),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cases::compressed_moves::BEFORE;
    use std::collections::BTreeSet;

    /// The cases name distinct recordings.
    #[test]
    fn the_formats_name_distinct_cases() {
        let named: BTreeSet<&str> = FORMATS.iter().map(|format| format.name).collect();
        assert_eq!(named.len(), FORMATS.len());
    }

    /// Every format's sibling is another format of the same block size, and the relation is its own
    /// inverse: the pairs, held to shape rather than taken on trust.
    #[test]
    fn every_sibling_is_a_pair_of_one_block_size() {
        for format in &FORMATS {
            let (raw, _) = format.sibling;
            let back = FORMATS
                .iter()
                .find(|other| other.format == raw)
                .unwrap_or_else(|| panic!("{} names no sibling in the table", format.name));
            assert_eq!(back.block, format.block, "{}", format.name);
            assert_eq!(back.sibling.0, format.format, "{}", format.name);
            assert_eq!(back.sibling.1, format.sibling.1, "{}", format.name);
            assert_eq!(sibling(format.format), Some(format.sibling));
        }
    }

    /// Every plan asks for exactly as many reads as it holds expectations, and no layout reaches
    /// past its value.
    #[test]
    fn every_plans_reads_match_its_expectations() {
        for case in all() {
            for (shape, plan) in case.plans() {
                assert_eq!(
                    plan.work.reads(),
                    plan.expected.len(),
                    "{} {}",
                    case.name(),
                    shape.name
                );
                for expected in &plan.expected {
                    assert!(
                        expected.layout.extent() <= expected.bytes.len(),
                        "{} {}/{}",
                        case.name(),
                        expected.step,
                        expected.leaf
                    );
                }
            }
        }
    }

    /// Every `out_tight` read is a buffer of blocks with no padding, so every one of its bytes is a
    /// block's: a read whose length is not the shape's tight blocks would be comparing something
    /// else.
    #[test]
    fn every_read_is_the_shapes_tight_blocks() {
        for case in all() {
            let block = case.format().block;
            for (shape, plan) in case.plans() {
                let tight: u64 = shape
                    .each()
                    .into_iter()
                    .map(|(l, _)| {
                        let [bx, by, bz] = shape.blocks(l);
                        bx * by * bz * block
                    })
                    .sum();
                for expected in &plan.expected {
                    assert_eq!(
                        expected.bytes.len() as u64,
                        tight,
                        "{} {}/{}",
                        case.name(),
                        shape.name,
                        expected.step
                    );
                }
            }
        }
    }

    /// Every one of Metal's recorded read-backs is the byte model's, byte for byte.
    ///
    /// The recordings hold what Metal answered on an Apple M3 Max; the model's other tie to Metal
    /// is one digest per step an Apple M4 printed. A difference is a finding either way — Apple's
    /// copies moved, or the model was never Metal's — and never something to edit away.
    #[test]
    fn every_recorded_view_move_is_the_byte_model() {
        let mut unrecorded = Vec::new();
        let mut differ = Vec::new();
        let mut checked = 0usize;
        for case in all() {
            let recording = match crate::recording::read(case.name()) {
                Ok(recording) => recording,
                Err(_) => {
                    unrecorded.push(case.name());
                    continue;
                }
            };
            let plans = case.plans();
            for answer in &recording.answers {
                for outcome in &answer.outcomes {
                    for (shape, plan) in &plans {
                        for expected in &plan.expected {
                            let path =
                                format!("{}/{}/{}", shape.name, expected.step, expected.leaf);
                            let node = outcome
                                .transcript
                                .root
                                .child(shape.name)
                                .and_then(|node| node.child(expected.step))
                                .and_then(|node| node.child(&expected.leaf));
                            let Some(node) = node else {
                                differ.push(format!("{}: no site {path}", case.name()));
                                continue;
                            };
                            let crate::transcript::Value::Bytes(bytes) = &node.observed.value
                            else {
                                differ.push(format!("{}/{path}: no bytes", case.name()));
                                continue;
                            };
                            checked += bytes.len();
                            if bytes != &expected.bytes && differ.len() < 24 {
                                let at =
                                    bytes.iter().zip(&expected.bytes).position(|(a, b)| a != b);
                                differ.push(format!(
                                    "{}/{path}: {} bytes against the model's {}, first difference \
                                     at {at:?}",
                                    case.name(),
                                    bytes.len(),
                                    expected.bytes.len(),
                                ));
                            }
                        }
                    }
                }
            }
        }
        assert!(
            differ.is_empty(),
            "{checked} bytes checked; Metal's recorded view copies differ from the byte model:\n{}",
            differ.join("\n")
        );
        assert!(
            unrecorded.is_empty(),
            "no recording yet for {unrecorded:?}: run `conformance record` on a Mac"
        );
    }

    /// `BEFORE` is the byte every read-back's buffer starts from, shared with the first lane's
    /// cases: a position no block reached is a recorded value and not whatever was there.
    #[test]
    fn the_before_byte_is_not_a_block_byte_of_any_plan() {
        // Every plan's blocks come from a splitmix64 stream, so a whole read-back of `BEFORE` would
        // mean no block moved at all. The guard is that at least one byte of every expectation
        // differs from it.
        for case in all() {
            for (shape, plan) in case.plans() {
                for expected in &plan.expected {
                    assert!(
                        expected.bytes.iter().any(|&byte| byte != BEFORE),
                        "{} {}/{}: the model expects nothing but the fill",
                        case.name(),
                        shape.name,
                        expected.step
                    );
                }
            }
        }
    }
}
