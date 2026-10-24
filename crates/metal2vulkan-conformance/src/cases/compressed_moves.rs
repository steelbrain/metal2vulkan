//! How block-compressed textures move their blocks, ported from
//! `runtime_lane::compressed_moves`'s first lane.
//!
//! The consumer's x86 guests hold BC textures, and corrupt texels on NVIDIA were what made this a
//! lane. Its subject is every BC format's blocks moving through shapes whose levels end in partial
//! blocks: a 2D texture 13 x 7 of four levels, a 2D array 10 x 6 of three slices and two levels, a
//! cube 12 x 12 of three levels, a 3D texture 9 x 5 x 3 of two levels, and a 2D texture 8 x 8
//! whose last two levels are smaller than a block. Per shape, five steps:
//!
//! - `put`: `replaceRegion:` of every level and slice whole, rows padded two blocks past their
//!   blocks and a 3D image one block past its rows, then one block replaced alone at block (1, 0)
//!   of level 0; read back by `getBytes:` of every level and slice whole, rows three blocks past
//!   and a 3D image two blocks past, into bytes of `0xcd`;
//! - `getsub`: `getBytes:` of each level wider than a block, from texel x 4 to its edges;
//! - `out`: a blit of every level and slice into one buffer of `0xcd`, each five blocks past the
//!   last, rows one block past their blocks;
//! - `in`: a blit of a buffer into a second, private texture, each level and slice three blocks
//!   past the last, rows two blocks past, then that texture blitted out tight;
//! - `t2t`: the first texture copied whole into a third, then a block-aligned region of the second
//!   copied over part of it, then the third blitted out tight.
//!
//! **What the port changes, and why it is the point.** The lane compares our bytes with a byte
//! model written in Rust ([`Model`]), and ties that model to Metal by a **single SHA-256 of a line
//! transcript** — 350 lines of `"<format> <shape> <step>: <n> bytes, sha <six bytes>"`, measured by
//! our own probe on an Apple **M4**. So the lane could say that our bytes and the model disagree,
//! never which of the two had moved, and its one Metal fact was a six-byte digest per step from a
//! machine we no longer have. Here record mode runs the same steps on **this M3 Max** and each
//! recording holds Metal's own bytes, padding included, per level and per slice. The model stays as
//! a cross-check (the test `every_recorded_move_is_the_lanes_byte_model`), which is the one direction in
//! which a difference says something: either Apple's copies moved between the chips or the releases,
//! or the model was never Metal's.
//!
//! **One case per format**, so a venue that holds no texture of one answers its typed refusal there
//! and the suite counts that format as [`LacksFeature`](crate::verdict::Verdict::LacksFeature). The
//! lane walks fourteen formats in one test, where a device missing one would take the whole lane
//! down. The port also covers BC6H and BC7, which no other case in the suite reaches.
//!
//! Every byte of every read-back is compared, the padding between the rows included
//! ([`Gaps::Compared`]): the lane asserts the bytes past each row's blocks stay `0xcd`, and here
//! that is the recorded value rather than an assertion beside it.
//!
//! **Not ported, and why.** The lane's second test adds a 2D texture of 1030 x 520 and all eleven
//! levels, whose level 0 alone is over 500 KiB of blocks — past the recording's whole-value ceiling,
//! so it wants the digest-plus-window form no case has exercised yet. Its other three shapes'
//! worth is covered by [`crate::cases::compressed_views`]. Stated here rather than left implicit.
//!
//! Nothing here can fault a GPU: host copies and blit copies between textures the program makes, no
//! dispatch, no draw, no loop, and every region block-aligned and inside its level.

use crate::allowance::Allowance;
use crate::case::{made, Abort, Case, Failure, Input, MetalSource, Side};
use crate::texture_query::Storage;
use crate::transcript::{Aspect, ByteLayout, Element, Gaps, Node, Observed, Region, Sites, Value};
use crate::work::{Plane, Rect, Resource, Step as WorkStep, TextureKind, TextureSpec, Work};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan_runtime::Degradation;

/// The byte a read-back's buffer holds before the copy, so a byte no block reached is a recorded
/// value and not whatever was there.
pub const BEFORE: u8 = 0xcd;

/// One BC format: the case's name, the raw `MTLPixelFormat` and its block's bytes.
struct Format {
    name: &'static str,
    /// The raw `MTLPixelFormat`, as `metal2vulkan::metal::PixelFormat` numbers it.
    /// Each row states it as `PixelFormat::<name>.as_raw()` rather than as a number, so
    /// the format a case puts to Metal is named in the case's own text and not only in a
    /// comment beside it.
    format: u32,
    block: u64,
}

/// Every BC format, in the lane's order, because a shape's seed is its place in that order and the
/// bytes must stay the bytes the lane measured.
const FORMATS: [Format; 14] = [
    Format {
        name: "compressed-moves-bc1-rgba",
        format: PixelFormat::BC1RGBA.as_raw(),
        block: 8,
    },
    Format {
        name: "compressed-moves-bc1-rgba-srgb",
        format: PixelFormat::BC1RGBASRGB.as_raw(),
        block: 8,
    },
    Format {
        name: "compressed-moves-bc2-rgba",
        format: PixelFormat::BC2RGBA.as_raw(),
        block: 16,
    },
    Format {
        name: "compressed-moves-bc2-rgba-srgb",
        format: PixelFormat::BC2RGBASRGB.as_raw(),
        block: 16,
    },
    Format {
        name: "compressed-moves-bc3-rgba",
        format: PixelFormat::BC3RGBA.as_raw(),
        block: 16,
    },
    Format {
        name: "compressed-moves-bc3-rgba-srgb",
        format: PixelFormat::BC3RGBASRGB.as_raw(),
        block: 16,
    },
    Format {
        name: "compressed-moves-bc4-r-unorm",
        format: PixelFormat::BC4RUnorm.as_raw(),
        block: 8,
    },
    Format {
        name: "compressed-moves-bc4-r-snorm",
        format: PixelFormat::BC4RSnorm.as_raw(),
        block: 8,
    },
    Format {
        name: "compressed-moves-bc5-rg-unorm",
        format: PixelFormat::BC5RGUnorm.as_raw(),
        block: 16,
    },
    Format {
        name: "compressed-moves-bc5-rg-snorm",
        format: PixelFormat::BC5RGSnorm.as_raw(),
        block: 16,
    },
    Format {
        name: "compressed-moves-bc6h-rgb-float",
        format: PixelFormat::BC6HRGBFloat.as_raw(),
        block: 16,
    },
    Format {
        name: "compressed-moves-bc6h-rgb-ufloat",
        format: PixelFormat::BC6HRGBUfloat.as_raw(),
        block: 16,
    },
    Format {
        name: "compressed-moves-bc7-rgba-unorm",
        format: PixelFormat::BC7RGBAUnorm.as_raw(),
        block: 16,
    },
    Format {
        name: "compressed-moves-bc7-rgba-unorm-srgb",
        format: PixelFormat::BC7RGBAUnormSRGB.as_raw(),
        block: 16,
    },
];

/// A texture's shape: its kind, its size in texels (the third axis a 3D texture's depth), its array
/// length and its levels. The lane's five, unchanged, because a shape's bytes come from a seeded
/// stream keyed to its place.
#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub name: &'static str,
    pub kind: TextureKind,
    pub size: [u64; 3],
    /// `arrayLength`; a cube's is 1 and its faces are its slices.
    pub array: u64,
    pub levels: u64,
}

/// The lane's five shapes, in its order.
pub const SHAPES: [Shape; 5] = [
    Shape {
        name: "2d-13x7",
        kind: TextureKind::Type2D,
        size: [13, 7, 1],
        array: 1,
        levels: 4,
    },
    Shape {
        name: "array-10x6x3",
        kind: TextureKind::Type2DArray,
        size: [10, 6, 1],
        array: 3,
        levels: 2,
    },
    Shape {
        name: "cube-12",
        kind: TextureKind::Cube,
        size: [12, 12, 1],
        array: 1,
        levels: 3,
    },
    Shape {
        name: "3d-9x5x3",
        kind: TextureKind::Type3D,
        size: [9, 5, 3],
        array: 1,
        levels: 2,
    },
    Shape {
        name: "2d-8x8",
        kind: TextureKind::Type2D,
        size: [8, 8, 1],
        array: 1,
        levels: 4,
    },
];

impl Shape {
    /// The slices a copy may name: a cube's six faces, otherwise the array length.
    pub fn slices(&self) -> u64 {
        match self.kind {
            TextureKind::Cube => 6,
            _ => self.array,
        }
    }

    /// Whether the shape is a 3D texture, whose copies carry a `bytesPerImage`.
    pub fn three_d(&self) -> bool {
        self.kind == TextureKind::Type3D
    }

    /// Level `l`'s size in texels.
    pub fn level(&self, l: u64) -> [u64; 3] {
        let [w, h, d] = self.size;
        [(w >> l).max(1), (h >> l).max(1), (d >> l).max(1)]
    }

    /// Level `l`'s size in blocks: its texels rounded up to 4 x 4, its depth as it is.
    pub fn blocks(&self, l: u64) -> [u64; 3] {
        let [w, h, d] = self.level(l);
        [w.div_ceil(4), h.div_ceil(4), d]
    }

    /// Every (level, slice), levels outermost, as the lane walks them.
    pub fn each(&self) -> Vec<(u64, u64)> {
        (0..self.levels)
            .flat_map(|l| (0..self.slices()).map(move |s| (l, s)))
            .collect()
    }

    /// The `bytesPerImage` of a level `l` copy whose rows are `row` apart, `extra` past its rows:
    /// a 3D texture's own images, 0 for a 2D texture, and a slice's rows for an array or a cube
    /// (which Metal takes but which names one image).
    pub fn image(&self, l: u64, row: u64, extra: u64) -> u64 {
        match self.kind {
            TextureKind::Type3D => row * self.blocks(l)[1] + extra,
            TextureKind::Type2D => 0,
            _ => row * self.blocks(l)[1],
        }
    }

    /// The texture the shape names, in `format`, private where `private`.
    pub fn spec(&self, format: u32, private: bool) -> TextureSpec {
        let [w, h, d] = self.size;
        let mut spec = TextureSpec::texture_2d(format, w, h).levels(self.levels);
        spec = match self.kind {
            TextureKind::Cube => spec.cube(w),
            TextureKind::Type3D => spec.volume(d),
            TextureKind::Type2DArray => spec.array(self.array),
            _ => spec,
        };
        if private {
            spec
        } else {
            spec.shared()
        }
    }
}

/// A splitmix64 stream of bytes: the lane's own, so a shape's blocks are the lane's blocks.
pub struct Rng(pub u64);

impl Rng {
    /// The next `n` bytes.
    pub fn bytes(&mut self, n: u64) -> Vec<u8> {
        (0..n)
            .map(|_| {
                self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = self.0;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                (z ^ (z >> 31)) as u8
            })
            .collect()
    }
}

/// A texture's blocks as Metal's copies leave them: `(level, slice, z, y, x)` to the block's bytes.
///
/// It is the lane's model, kept whole, because the whole point of the port is to hold Metal's own
/// recorded bytes against it.
#[derive(Clone, Default)]
pub struct Model(std::collections::BTreeMap<(u64, u64, u64, u64, u64), Vec<u8>>);

impl Model {
    /// The blocks of `count` (x, y, z) at block `at` of `(l, s)` read out of `bytes` at `offset`,
    /// rows `row` and images `image` apart.
    pub fn take(
        &mut self,
        (l, s): (u64, u64),
        at: [u64; 3],
        count: [u64; 3],
        (bytes, offset, row, image): (&[u8], u64, u64, u64),
        block: u64,
    ) {
        for z in 0..count[2] {
            for y in 0..count[1] {
                for x in 0..count[0] {
                    let from = (offset + z * image + y * row + x * block) as usize;
                    let key = (l, s, at[2] + z, at[1] + y, at[0] + x);
                    self.0
                        .insert(key, bytes[from..from + block as usize].to_vec());
                }
            }
        }
    }

    /// The blocks of `count` at block `at` of `(l, s)` written over `bytes` at `offset`, rows `row`
    /// and images `image` apart.
    pub fn give(
        &self,
        (l, s): (u64, u64),
        at: [u64; 3],
        count: [u64; 3],
        (bytes, offset, row, image): (&mut [u8], u64, u64, u64),
        block: u64,
    ) {
        for z in 0..count[2] {
            for y in 0..count[1] {
                for x in 0..count[0] {
                    let to = (offset + z * image + y * row + x * block) as usize;
                    let key = (l, s, at[2] + z, at[1] + y, at[0] + x);
                    bytes[to..to + block as usize].copy_from_slice(&self.0[&key]);
                }
            }
        }
    }

    /// `other`'s blocks of `count` at `from` of `(l, s)` copied to `to` of `(l2, s2)`.
    pub fn copy(
        &mut self,
        other: &Model,
        ((l, s), from): ((u64, u64), [u64; 3]),
        ((l2, s2), to): ((u64, u64), [u64; 3]),
        count: [u64; 3],
    ) {
        for z in 0..count[2] {
            for y in 0..count[1] {
                for x in 0..count[0] {
                    let block = other.0[&(l, s, from[2] + z, from[1] + y, from[0] + x)].clone();
                    self.0
                        .insert((l2, s2, to[2] + z, to[1] + y, to[0] + x), block);
                }
            }
        }
    }
}

/// The steps, in the lane's order, each naming the site its read-backs sit under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Step {
    Put,
    GetSub,
    Out,
    In,
    TextureToTexture,
}

impl Step {
    /// The site name the step's reads sit under.
    const fn site(self) -> &'static str {
        match self {
            Step::Put => "put",
            Step::GetSub => "getsub",
            Step::Out => "out",
            Step::In => "in",
            Step::TextureToTexture => "t2t",
        }
    }
}

/// One read-back the program asks for: where it sits in the transcript, what the model says it
/// should hold, and the layout the comparison reads it under.
pub struct Expected {
    /// The step the read belongs to, which is the branch it sits under.
    pub step: &'static str,
    /// The leaf's name inside the step: `<level>-<slice>` for a per-level read, `buffer` for a
    /// whole buffer read out tight.
    pub leaf: String,
    pub bytes: Vec<u8>,
    pub layout: ByteLayout,
}

/// A shape's program and what the model expects each of its reads to hold, in the order the program
/// asks for them.
pub struct Plan {
    pub work: Work,
    pub expected: Vec<Expected>,
}

/// The layout a read-back of `(l, s)` of `shape` is compared under: `across` blocks of `block`
/// bytes in a row, rows `row` apart, `down` rows an image, images `image` apart, every byte
/// compared including the padding past each row.
///
/// The region it names is the **texels** the call read, which is not the blocks rounded up: a level
/// 13 texels wide is four blocks across, and a diff should name the level's own texels.
fn texture_layout(
    (l, s): (u64, u64),
    [across, down, deep]: [u64; 3],
    (block, row, image): (u64, u64, u64),
    (origin, size): ([u64; 3], [u64; 3]),
) -> ByteLayout {
    ByteLayout {
        region: Region::Texture {
            level: l as u32,
            slice: s as u32,
            aspect: Aspect::Color,
            sample: 0,
            origin: [origin[0] as u32, origin[1] as u32, origin[2] as u32],
            size: [size[0] as u32, size[1] as u32, size[2] as u32],
        },
        element: Element::U8,
        row_offset: 0,
        row_elements: (across * block) as u32,
        row_pitch: row as u32,
        rows: down as u32,
        image_pitch: if image == 0 {
            (row * down) as u32
        } else {
            image as u32
        },
        images: deep as u32,
        gaps: Gaps::Compared,
    }
}

/// The plan for `format` in `shape`, seeded as the lane seeds it: the whole five-step program and
/// the model's bytes for each read.
///
/// The one function builds both, so the program's offsets and the model's offsets cannot drift
/// apart — the failure mode the formats batch hit twice was two sides of one shape written twice.
fn plan(format: u32, block: u64, shape: &Shape, seed: u64) -> Plan {
    let mut rng = Rng(seed);
    let mut work = Work::new();
    let mut expected = Vec::new();
    let three_d = shape.three_d();
    let first = work.add(Resource::Texture {
        spec: shape.spec(format, false),
    });
    let second = work.add(Resource::Texture {
        spec: shape.spec(format, true),
    });
    let third = work.add(Resource::Texture {
        spec: shape.spec(format, true),
    });
    let mut model = Model::default();

    // put: every level and slice whole, rows two blocks past, then one block alone at block (1, 0)
    // of level 0, slice 0.
    for (l, s) in shape.each() {
        let [w, h, d] = shape.level(l);
        let [bx, by, bz] = shape.blocks(l);
        let row = bx * block + 2 * block;
        let image = if three_d { row * by + block } else { 0 };
        let total = if three_d { image * bz } else { row * by };
        let bytes = rng.bytes(total);
        model.take((l, s), [0; 3], [bx, by, bz], (&bytes, 0, row, image), block);
        work.step(WorkStep::ReplaceRegion {
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
    if shape.blocks(0)[0] >= 2 {
        let h = shape.level(0)[1].min(4);
        let bytes = rng.bytes(block);
        let image = if three_d { block } else { 0 };
        model.take(
            (0, 0),
            [1, 0, 0],
            [1, 1, 1],
            (&bytes, 0, block, image),
            block,
        );
        work.step(WorkStep::ReplaceRegion {
            texture: first,
            level: 0,
            slice: 0,
            region: Rect::new_3d([4, 0, 0], [4, h, 1]),
            bytes,
            rows: 1,
            bytes_per_row: block,
            bytes_per_image: image,
        });
    }
    // The read-back: every level and slice whole, rows three blocks past, a 3D image two past.
    for (l, s) in shape.each() {
        let [w, h, d] = shape.level(l);
        let [bx, by, bz] = shape.blocks(l);
        let row = bx * block + 3 * block;
        let image = if three_d { row * by + 2 * block } else { 0 };
        let total = if three_d { image * bz } else { row * by };
        let mut want = vec![BEFORE; total as usize];
        model.give(
            (l, s),
            [0; 3],
            [bx, by, bz],
            (&mut want, 0, row, image),
            block,
        );
        expected.push(Expected {
            step: Step::Put.site(),
            leaf: format!("{l}-{s}"),
            bytes: want,
            layout: texture_layout(
                (l, s),
                [bx, by, bz],
                (block, row, image),
                ([0, 0, 0], [w, h, d]),
            ),
        });
        work.step(WorkStep::GetBytes {
            texture: first,
            level: l,
            slice: s,
            region: Rect::new_3d([0, 0, 0], [w, h, d]),
            bytes_per_row: row,
            bytes_per_image: image,
            length: total as usize,
            // The lane reads into bytes of 0xcd, so the bytes past each row's blocks are recorded
            // as a value Metal would never write: that is what makes "the padding stayed as it
            // was" a question with an answer.
            before: BEFORE,
        });
    }

    // getsub: from texel x 4 to each level's edges, where the level is wider than a block.
    for (l, s) in shape.each() {
        let [w, h, d] = shape.level(l);
        let [bx, by, bz] = shape.blocks(l);
        if w <= 4 {
            continue;
        }
        let across = bx - 1;
        let row = across * block + block;
        let image = if three_d { row * by + block } else { 0 };
        let total = if three_d { image * bz } else { row * by };
        let mut want = vec![BEFORE; total as usize];
        model.give(
            (l, s),
            [1, 0, 0],
            [across, by, bz],
            (&mut want, 0, row, image),
            block,
        );
        expected.push(Expected {
            step: Step::GetSub.site(),
            leaf: format!("{l}-{s}"),
            bytes: want,
            layout: texture_layout(
                (l, s),
                [across, by, bz],
                (block, row, image),
                ([4, 0, 0], [w - 4, h, d]),
            ),
        });
        work.step(WorkStep::GetBytes {
            texture: first,
            level: l,
            slice: s,
            region: Rect::new_3d([4, 0, 0], [w - 4, h, d]),
            bytes_per_row: row,
            bytes_per_image: image,
            length: total as usize,
            // The lane reads into bytes of 0xcd, so the bytes past each row's blocks are recorded
            // as a value Metal would never write: that is what makes "the padding stayed as it
            // was" a question with an answer.
            before: BEFORE,
        });
    }

    // out: every level and slice into one buffer of BEFORE, each five blocks past the last, rows
    // one block past their blocks.
    let mut places = Vec::new();
    let mut at = 0;
    for (l, s) in shape.each() {
        let [bx, by, bz] = shape.blocks(l);
        let row = bx * block + block;
        let image = shape.image(l, row, block);
        at += 5 * block;
        places.push((l, s, at, row, image));
        at += if three_d { image * bz } else { row * by };
    }
    let out = work.buffer(vec![BEFORE; at as usize]);
    for &(l, s, offset, row, image) in &places {
        let [w, h, d] = shape.level(l);
        work.step(WorkStep::CopyTextureToBuffer {
            texture: first,
            slice: s,
            level: l,
            origin: [0, 0, 0],
            size: [w, h, d],
            buffer: out,
            offset,
            bytes_per_row: row,
            bytes_per_image: image,
            plane: Plane::Whole,
        });
    }
    work.step(WorkStep::Commit);
    let mut want = vec![BEFORE; at as usize];
    for &(l, s, offset, row, image) in &places {
        let image = if three_d { image } else { 0 };
        model.give(
            (l, s),
            [0; 3],
            shape.blocks(l),
            (&mut want, offset, row, image),
            block,
        );
    }
    expected.push(Expected {
        step: Step::Out.site(),
        leaf: "buffer".to_owned(),
        bytes: want,
        layout: ByteLayout::buffer(0, at, Element::U8),
    });
    work.step(WorkStep::ReadBuffer {
        buffer: out,
        offset: 0,
        length: at as usize,
    });

    // in: a buffer into the second texture, each level and slice three blocks past the last, rows
    // two blocks past; then that texture out tight.
    let mut places = Vec::new();
    let mut at = 0;
    for (l, s) in shape.each() {
        let [bx, by, bz] = shape.blocks(l);
        let row = bx * block + 2 * block;
        let image = shape.image(l, row, 2 * block);
        at += 3 * block;
        places.push((l, s, at, row, image));
        at += if three_d { image * bz } else { row * by };
    }
    let source = rng.bytes(at);
    let source_buffer = work.buffer(source.clone());
    for &(l, s, offset, row, image) in &places {
        let [w, h, d] = shape.level(l);
        work.step(WorkStep::CopyBufferToTexture {
            buffer: source_buffer,
            offset,
            bytes_per_row: row,
            bytes_per_image: image,
            size: [w, h, d],
            texture: second,
            slice: s,
            level: l,
            origin: [0, 0, 0],
            plane: Plane::Whole,
        });
    }
    work.step(WorkStep::Commit);
    let mut second_model = Model::default();
    for &(l, s, offset, row, image) in &places {
        let image = if three_d { image } else { 0 };
        second_model.take(
            (l, s),
            [0; 3],
            shape.blocks(l),
            (&source, offset, row, image),
            block,
        );
    }
    out_tight(
        &mut work,
        &mut expected,
        shape,
        block,
        second,
        &second_model,
        Step::In.site(),
    );

    // t2t: the first texture whole into a third, then a region of the second over part of it.
    let Moved {
        from,
        to,
        size,
        from_at,
        to_at,
    } = region(shape);
    for (l, s) in shape.each() {
        let [w, h, d] = shape.level(l);
        work.step(WorkStep::CopyTextureToTexture {
            from: first,
            from_slice: s,
            from_level: l,
            from_origin: [0, 0, 0],
            size: [w, h, d],
            to: third,
            to_slice: s,
            to_level: l,
            to_origin: [0, 0, 0],
        });
    }
    work.step(WorkStep::CopyTextureToTexture {
        from: second,
        from_slice: from_at.1,
        from_level: from_at.0,
        from_origin: from,
        size,
        to: third,
        to_slice: to_at.1,
        to_level: to_at.0,
        to_origin: to,
    });
    work.step(WorkStep::Commit);
    let mut third_model = Model::default();
    for (l, s) in shape.each() {
        third_model.copy(&model, ((l, s), [0; 3]), ((l, s), [0; 3]), shape.blocks(l));
    }
    let in_blocks = |v: [u64; 3]| [v[0] / 4, v[1] / 4, v[2]];
    let count = [size[0].div_ceil(4), size[1].div_ceil(4), size[2]];
    third_model.copy(
        &second_model,
        (from_at, in_blocks(from)),
        (to_at, in_blocks(to)),
        count,
    );
    out_tight(
        &mut work,
        &mut expected,
        shape,
        block,
        third,
        &third_model,
        Step::TextureToTexture.site(),
    );
    Plan { work, expected }
}

/// The texture-to-texture copy a shape's `t2t` step makes: where the blocks come from, where they
/// go, and how many.
struct Moved {
    /// The source origin in texels.
    from: [u64; 3],
    /// The destination origin in texels.
    to: [u64; 3],
    /// The size in texels.
    size: [u64; 3],
    /// The (level, slice) the source names.
    from_at: (u64, u64),
    /// The (level, slice) the destination names.
    to_at: (u64, u64),
}

/// The lane's region for `shape`, each chosen so the copy is block-aligned and reaches a level's
/// edge in at least one axis.
fn region(shape: &Shape) -> Moved {
    let (from, to, size, from_at, to_at) = match shape.name {
        // Blocks (1..3, 1) of level 0 to (0..2, 1): x interior, y to the edge.
        "2d-13x7" => ([4, 4, 0], [0, 4, 0], [8, 3, 1], (0, 0), (0, 0)),
        // Block (1, 1) of level 0 to (0, 0).
        "2d-8x8" => ([4, 4, 0], [0, 0, 0], [4, 4, 1], (0, 0), (0, 0)),
        // Level 1 of slice 2 whole to slice 0.
        "array-10x6x3" => ([0, 0, 0], [0, 0, 0], [5, 3, 1], (1, 2), (1, 0)),
        // Blocks (1..3, 0..3) of face 2 to face 5: to the edges.
        "cube-12" => ([4, 0, 0], [4, 0, 0], [8, 12, 1], (0, 2), (0, 5)),
        // Depth 1..3 of level 0 to depth 0..2.
        _ => ([0, 0, 1], [0, 0, 0], [9, 5, 2], (0, 0), (0, 0)),
    };
    Moved {
        from,
        to,
        size,
        from_at,
        to_at,
    }
}

/// Every level and slice of `texture` blitted out tight, one after another from byte 0, into a
/// buffer of [`BEFORE`], with the model's bytes recorded for it under `step`.
///
/// It is the lane's `out_tight`, which both compressed families use as their one way of reading a
/// private texture's blocks back.
pub fn out_tight(
    work: &mut Work,
    expected: &mut Vec<Expected>,
    shape: &Shape,
    block: u64,
    texture: usize,
    model: &Model,
    step: &'static str,
) {
    let three_d = shape.three_d();
    let mut places = Vec::new();
    let mut at = 0;
    for (l, s) in shape.each() {
        let [bx, by, bz] = shape.blocks(l);
        let row = bx * block;
        let image = shape.image(l, row, 0);
        places.push((l, s, at, row, image));
        at += row * by * bz;
    }
    let buffer = work.buffer(vec![BEFORE; at as usize]);
    for &(l, s, offset, row, image) in &places {
        let [w, h, d] = shape.level(l);
        work.step(WorkStep::CopyTextureToBuffer {
            texture,
            slice: s,
            level: l,
            origin: [0, 0, 0],
            size: [w, h, d],
            buffer,
            offset,
            bytes_per_row: row,
            bytes_per_image: image,
            plane: Plane::Whole,
        });
    }
    work.step(WorkStep::Commit);
    let mut want = vec![BEFORE; at as usize];
    for &(l, s, offset, row, image) in &places {
        let image = if three_d { image } else { 0 };
        model.give(
            (l, s),
            [0; 3],
            shape.blocks(l),
            (&mut want, offset, row, image),
            block,
        );
    }
    expected.push(Expected {
        step,
        leaf: "buffer".to_owned(),
        bytes: want,
        layout: ByteLayout::buffer(0, at, Element::U8),
    });
    work.step(WorkStep::ReadBuffer {
        buffer,
        offset: 0,
        length: at as usize,
    });
}

/// The notices a venue announces where it does not hold a BC format in its own compressed form, and
/// which a case may therefore name as justifying its notices site.
///
/// Measured per case, each run **alone** so the once-per-device attribution is right: on lavapipe
/// BC1 to BC5 announce `cpu-block-decode` and `decoded-block-twin` (the blocks live in a carrier and
/// its reads go through a decoded twin), BC7 announces `compressed-texel-decode`, and BC6H announces
/// nothing because lavapipe holds it natively. On MoltenVK none of them fires. Every block still
/// compares exactly on both: the only site these loosen is the set of notices, which Metal holds
/// empty, and an allowance fails the case if the notice it names did not fire.
///
/// `compressed-texel-decode` is named for **BC6H and BC7 only**.
/// [`crate::cases::compressed_decode`] deliberately refuses to allow it for BC1 to BC5, because its
/// lane asserts no texture of those formats announces it, and a run that does is a divergence to
/// report rather than one to excuse. That assertion does not cover BC6H or BC7, and lavapipe does
/// announce it for BC7, so naming it there keeps the check alive where it exists and tells the truth
/// where it does not.
///
/// What is **not** here: `staged-texture-copy`. Our `replaceRegion:` into a compressed texture does
/// go through a staging buffer, and the notice fires on **no** venue of this suite for these cases
/// (verified over four legs: MoltenVK and lavapipe, exact and default policy). A tolerance naming a
/// mechanism no venue announces is a claim nothing stands behind, so it is gone, as the `formats`
/// stream's `shadowed-texture` went.
pub fn carrier_allowances(format: u32) -> Vec<Allowance> {
    // Every row reads its answer back through a `Shared` buffer, so on a device without
    // unified memory the runtime announces `shared-in-system-memory`
    // (`crate::replay::shared_memory_notice` holds the whole reason). It declares the
    // notices site alone, so every value of this family still compares against Metal's
    // exactly.
    //
    // Measured on the RTX 5080 Laptop (615.71.09): it fired for
    // all 14 rows of each of the two families that call this, and each was divergent on that
    // field and nothing else (replay-nvidia.log). On MoltenVK, where
    // the chip's GPU shares the CPU's memory, it cannot fire at all, which is why no
    // recording of this suite declares it.
    let mut codes = vec![
        Degradation::SharedInSystemMemory.code(),
        Degradation::CpuBlockDecode.code(),
        Degradation::DecodedBlockTwin.code(),
    ];
    // BC6H (150, 151) and BC7 (152, 153); the BC1 to BC5 numbers below them are the family
    // `compressed_decode`'s own assertion covers.
    if format >= 150 {
        codes.push(Degradation::CompressedTexelDecode.code());
    }
    codes
        .into_iter()
        .map(crate::replay::notice_allowance)
        .collect()
}

/// The case for one format.
pub struct CompressedMoves {
    at: usize,
}

/// Every format's case, in the lane's order.
pub fn all() -> Vec<CompressedMoves> {
    (0..FORMATS.len())
        .map(|at| CompressedMoves { at })
        .collect()
}

impl CompressedMoves {
    /// The format the case moves.
    fn format(&self) -> &'static Format {
        &FORMATS[self.at]
    }

    /// The plans of every shape, in the lane's order, each seeded as the lane seeds it: the block
    /// size and the shape's place, so the bytes are the bytes the lane measured.
    pub fn plans(&self) -> Vec<(&'static Shape, Plan)> {
        let format = self.format();
        SHAPES
            .iter()
            .enumerate()
            .map(|(k, shape)| {
                let seed = 0x5eed_0000 + format.block * 0x100 + k as u64;
                (shape, plan(format.format, format.block, shape, seed))
            })
            .collect()
    }
}

impl Case for CompressedMoves {
    fn name(&self) -> &'static str {
        FORMATS[self.at].name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        carrier_allowances(self.format().format)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        // The whole program of every shape, as the interpreter will run it: every descriptor, every
        // region, every pitch and every byte written. A case whose program moves is `Unrecorded`
        // rather than compared against a recording of other bytes.
        let mut inputs = vec![Input::bytes("format", &self.format().format.to_le_bytes())];
        for (shape, plan) in self.plans() {
            inputs.push(Input::bytes(
                format!("program:{}", shape.name),
                &fingerprint(&plan.work),
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
                let node = Node::leaf(
                    expected.leaf.clone(),
                    Observed::bytes(bytes.to_vec(), expected.layout.clone()),
                );
                match nodes.last_mut() {
                    Some(last) if last.name == expected.step => last.children.push(node),
                    _ => nodes.push(Node::branch(expected.step.to_owned(), vec![node])),
                }
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

/// A program's fingerprint, keyed by each step's **Metal selector** and its own parameters rather
/// than by its position in the vocabulary.
///
/// The compressed families fingerprint their whole program, because a pitch or a region moving is
/// exactly the kind of change that must make a recording `Unrecorded` rather than compare against
/// Metal's answer over other bytes. Writing that as `format!("{work:?}")` would have worked and been
/// a trap: every later addition to [`crate::work`] — a field no case here sets — would move the
/// fingerprint and turn every one of these recordings stale at once, in a stream that is no longer
/// running. So each step contributes the name of the call it makes and the numbers and bytes it
/// hands that call, and a resource contributes its kind and its own shape. A new variant or a new
/// field then changes nothing for a case that does not use it.
pub fn fingerprint(work: &Work) -> Vec<u8> {
    let mut out = Vec::new();
    let word = |out: &mut Vec<u8>, value: u64| out.extend(value.to_le_bytes());
    for resource in &work.resources {
        match resource {
            Resource::Buffer { bytes } => {
                out.extend(b"newBufferWithBytes");
                word(&mut out, bytes.len() as u64);
                out.extend(bytes);
            }
            Resource::UnwrittenBuffer { length, storage } => {
                out.extend(b"newBufferWithLength");
                word(&mut out, *length);
                // The storage is emitted only where it is **not** `Private`, which is the single
                // mode the variant carried when these recordings were taken (it was
                // `Resource::PrivateBuffer`). So a private one fingerprints exactly as it did, and
                // a case that asks for another mode says so in its own bytes. No case of these
                // four families makes one at all, so the arm fires nowhere today.
                if *storage != Storage::Private {
                    out.extend(storage.word().as_bytes());
                }
            }
            Resource::Texture { spec } => {
                out.extend(b"newTextureWithDescriptor");
                out.extend(texture_words(spec));
            }
            Resource::View {
                texture,
                format,
                level,
                level_count,
                slice,
                slice_count,
                flatten,
            } => {
                out.extend(b"newTextureViewWithPixelFormat");
                for value in [
                    *texture as u64,
                    u64::from(*format),
                    *level,
                    *level_count,
                    *slice,
                    *slice_count,
                    u64::from(*flatten),
                ] {
                    word(&mut out, value);
                }
            }
            // No case in these families makes one, so its shape is whatever it is: a variant none
            // of them uses cannot move their fingerprints.
            other => out.extend(format!("{other:?}").as_bytes()),
        }
    }
    for step in &work.steps {
        match step {
            WorkStep::ReplaceRegion {
                texture,
                level,
                slice,
                region,
                bytes,
                rows,
                bytes_per_row,
                bytes_per_image,
            } => {
                out.extend(b"replaceRegion");
                for value in [
                    *texture as u64,
                    *level,
                    *slice,
                    *rows,
                    *bytes_per_row,
                    *bytes_per_image,
                ] {
                    word(&mut out, value);
                }
                out.extend(rect_words(region));
                word(&mut out, bytes.len() as u64);
                out.extend(bytes);
            }
            WorkStep::GetBytes {
                texture,
                level,
                slice,
                region,
                bytes_per_row,
                bytes_per_image,
                length,
                before,
            } => {
                out.extend(b"getBytes");
                for value in [
                    *texture as u64,
                    *level,
                    *slice,
                    *bytes_per_row,
                    *bytes_per_image,
                    *length as u64,
                    u64::from(*before),
                ] {
                    word(&mut out, value);
                }
                out.extend(rect_words(region));
            }
            WorkStep::CopyBufferToTexture {
                buffer,
                offset,
                bytes_per_row,
                bytes_per_image,
                size,
                texture,
                slice,
                level,
                origin,
                plane,
            } => {
                out.extend(b"copyFromBufferToTexture");
                for value in [
                    *buffer as u64,
                    *offset,
                    *bytes_per_row,
                    *bytes_per_image,
                    *texture as u64,
                    *slice,
                    *level,
                ] {
                    word(&mut out, value);
                }
                out.extend(rect_words(&Rect::new_3d(*origin, *size)));
                out.extend(format!("{plane:?}").as_bytes());
            }
            WorkStep::CopyTextureToBuffer {
                texture,
                slice,
                level,
                origin,
                size,
                buffer,
                offset,
                bytes_per_row,
                bytes_per_image,
                plane,
            } => {
                out.extend(b"copyFromTextureToBuffer");
                for value in [
                    *texture as u64,
                    *slice,
                    *level,
                    *buffer as u64,
                    *offset,
                    *bytes_per_row,
                    *bytes_per_image,
                ] {
                    word(&mut out, value);
                }
                out.extend(rect_words(&Rect::new_3d(*origin, *size)));
                out.extend(format!("{plane:?}").as_bytes());
            }
            WorkStep::CopyTextureToTexture {
                from,
                from_slice,
                from_level,
                from_origin,
                size,
                to,
                to_slice,
                to_level,
                to_origin,
            } => {
                out.extend(b"copyFromTextureToTexture");
                for value in [
                    *from as u64,
                    *from_slice,
                    *from_level,
                    *to as u64,
                    *to_slice,
                    *to_level,
                ] {
                    word(&mut out, value);
                }
                out.extend(rect_words(&Rect::new_3d(*from_origin, *size)));
                out.extend(rect_words(&Rect::new_3d(*to_origin, *size)));
            }
            WorkStep::ReadBuffer {
                buffer,
                offset,
                length,
            } => {
                out.extend(b"contents");
                for value in [*buffer as u64, *offset, *length as u64] {
                    word(&mut out, value);
                }
            }
            WorkStep::Commit => out.extend(b"commit"),
            // As above: a step none of these families encodes cannot move their fingerprints.
            other => out.extend(format!("{other:?}").as_bytes()),
        }
    }
    out
}

/// A texture's own shape, as the fields the compressed families set.
fn texture_words(spec: &TextureSpec) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend(format!("{:?}", spec.kind).as_bytes());
    for value in [
        u64::from(spec.format),
        spec.width,
        spec.height,
        spec.depth,
        spec.levels,
        spec.slices,
        spec.samples,
        // `matches!(.., Private)` rather than the mode's own ordinal, deliberately: these
        // recordings were taken when the field was a `private: bool`, and one byte of 0 or 1 is
        // what their fingerprints hold. Writing the ordinal instead would make every case with a
        // private texture `Unrecorded` for a rename that changed no program. None of these
        // families asks for `Managed`, which
        // `no_texture_of_these_families_asks_for_managed_storage` holds, so this byte is the byte
        // that was recorded; the day one does, that guard fires rather than a recording going
        // quietly stale.
        u64::from(matches!(spec.storage, Storage::Private)),
        u64::from(spec.gpu_optimized),
        u64::from(spec.usage.unwrap_or(0)),
        u64::from(spec.usage.is_some()),
    ] {
        out.extend(value.to_le_bytes());
    }
    out
}

/// A region's origin and size, in texels.
fn rect_words(rect: &Rect) -> Vec<u8> {
    let mut out = Vec::new();
    for axis in rect.origin.iter().chain(&rect.size) {
        out.extend(axis.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    /// The cases name distinct recordings, since a case's name is its recording's file name.
    #[test]
    fn the_formats_name_distinct_cases() {
        let named: BTreeSet<&str> = FORMATS.iter().map(|format| format.name).collect();
        assert_eq!(named.len(), FORMATS.len());
    }

    /// Every plan asks for exactly as many reads as it holds expectations, and every expectation's
    /// layout covers its bytes: the two halves of one plan agree by construction.
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
                    // A padded read-back's last row keeps its padding, which the layout's extent
                    // does not count: the extent reaches the last element, and the value holds the
                    // bytes past it too, which `Gaps::Compared` is what compares.
                    assert!(
                        expected.layout.extent() <= expected.bytes.len(),
                        "{} {}/{}: a layout reaching {} bytes over a value of {}",
                        case.name(),
                        expected.step,
                        expected.leaf,
                        expected.layout.extent(),
                        expected.bytes.len(),
                    );
                }
            }
        }
    }

    /// No texture of these four families asks for `Managed` storage.
    ///
    /// This is not tidiness, it is what makes [`fingerprint`] honest. The 42 recordings were taken
    /// when `TextureSpec` carried a `private: bool`, so their fingerprints hold one byte of 0 or 1
    /// for a texture's storage. `fingerprint` therefore writes
    /// `matches!(spec.storage, Storage::Private)`, which is byte-identical to that **only while no
    /// case uses the third mode**. The day one does, this fails and says so, instead of a recording
    /// going quietly stale or — worse — a `Managed` texture fingerprinting as a `Shared` one.
    #[test]
    fn no_texture_of_these_families_asks_for_managed_storage() {
        let mut specs = 0usize;
        let mut works: Vec<Work> = Vec::new();
        for case in all() {
            works.extend(case.plans().into_iter().map(|(_, plan)| plan.work));
        }
        for case in crate::cases::compressed_views::all() {
            works.extend(case.plans().into_iter().map(|(_, plan)| plan.work));
        }
        for case in crate::cases::pvrtc_rows::all() {
            works.push(case.plan().work);
        }
        for case in crate::cases::pvrtc_cpu_order::all() {
            works.push(case.plan().work);
        }
        for work in &works {
            for resource in &work.resources {
                if let Resource::Texture { spec }
                | Resource::HeapForTexture { spec, .. }
                | Resource::HeapTexture { spec, .. }
                | Resource::TextureOverBuffer { spec, .. } = resource
                {
                    specs += 1;
                    assert_ne!(
                        spec.storage,
                        Storage::Managed,
                        "a texture of these families asks for Managed storage, so \
                         `fingerprint`'s one-byte storage is no longer the byte that was recorded"
                    );
                }
                assert!(
                    !matches!(resource, Resource::UnwrittenBuffer { .. }),
                    "a case of these families now makes an unwritten buffer; check its \
                     fingerprint bytes against the recording before trusting this run"
                );
            }
        }
        assert!(specs >= 150, "only {specs} texture specs reached the guard");
    }

    /// Every one of Metal's recorded read-backs is the lane's byte model's, byte for byte, padding
    /// included.
    ///
    /// This is the cross-check the port exists for. The recordings hold what Metal answered on this
    /// M3 Max; the model was tied to Metal only by a six-byte digest per step, measured by our own
    /// probe on an Apple M4. A difference means one of two things, and either is a finding rather
    /// than something to edit away: Apple's copies moved between the chips or the releases, or the
    /// model was never Metal's.
    ///
    /// A case with no recording yet is named, so a missing recording is record mode's work and not
    /// a silent pass.
    #[test]
    fn every_recorded_move_is_the_lanes_byte_model() {
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
            "{checked} bytes checked; Metal's recorded copies differ from the lane's byte \
             model:\n{}",
            differ.join("\n")
        );
        assert!(
            unrecorded.is_empty(),
            "no recording yet for {unrecorded:?}: run `conformance record` on a Mac"
        );
    }
}
