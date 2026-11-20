//! A view of another texture type over the same texture, as Metal casts one
//! (`newTextureViewWithPixelFormat:textureType:levels:slices:`): rendered into, the parent sampled,
//! and the parent rendered into, the view sampled.
//!
//! | case | the parent | the view: type, levels, slices |
//! |---|---|---|
//! | `texture-view-casts-cube-as-2d-array-whole` | a cube | a 2D array of every level and all six faces, the replay streams' form |
//! | `texture-view-casts-cube-as-2d-array-faces` | a cube | a 2D array of levels 1 and 2, faces 2 to 4 |
//! | `texture-view-casts-cube-as-2d` | a cube | a 2D texture of level 1, face 4 |
//! | `texture-view-casts-2d-as-2d-array` | a 2D texture | a 2D array of levels 1 and 2, its one slice |
//! | `texture-view-casts-2d-array-as-cube` | a 2D array of eight slices | a cube of levels 1 and 2, slices 1 to 6 |
//!
//! These are the casts of Apple's table (`makeTextureView(pixelFormat:textureType:levels:slices:)`)
//! that metal2vulkan-runtime builds of another type than the parent's: those a cube or a 2D texture
//! makes with nothing new at its make, and a square 2D array's cube of six slices, whose image rt
//! makes cube-compatible; a cube array, and a view as one, is a gap of its own there.
//!
//! Every parent is `RGBA16Float`, 8 by 8 texels at level 0, of three levels, `Private`, of
//! render-target and shader-read usage; a cube's slices are its faces, as Metal counts a view's
//! `sliceRange`, and a 2D array has eight slices. Each case makes the parent and
//! the view, and records the view's `textureType`, `pixelFormat`, `mipmapLevelCount` and
//! `arrayLength`. Then, twice:
//!
//! - **into the view**: one render pass a level and slice of the view, each clearing its
//!   attachment to a colour that names the level and slice, then one dispatch that samples every
//!   texel of the parent's levels and slices under the view;
//! - **into the parent**: the same over every level and slice of the parent, then a dispatch that
//!   samples every texel of every level and slice of the view.
//!
//! A sample is the owned fixture `tvc_samples`'s: a nearest sample at an explicit level, at the
//! texel's centre, of a 2D texture, a 2D array or a cube (along the direction through the texel's
//! centre, by the face table Metal and Vulkan share), whichever the sampled texture is. A colour
//! names its level and slice in sixteenths and eighths, which a half holds exactly, so each sample
//! is the colour cleared where the view and the parent share the texel, or it is a difference.

use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Case, Failure, Handle, Input, MetalSource, Side};
use crate::fixture::{Fixture, METAL_TEXT, RUNTIME_TEXT};
use crate::recording::ShaderRef;
use crate::render::{AttachmentPlace, Dispatch, PassColor, RenderPass, RenderWork};
use crate::texture::TextureRequest;
use crate::texture_views::{ViewCall, ViewRequest};
use crate::transcript::{ByteLayout, Digest, Element, Gaps, Node, Observed, Region, Sites, Value};
use metal2vulkan::metal::{PixelFormat, TextureSwizzleChannels, TextureType, TextureUsage};
use TextureType::{Cube, Type2D, Type2DArray};

/// The owned fixture of the three samplers.
const FIXTURE: &str = "tvc_samples";

/// Every parent's and view's pixel format: the replay streams' cube's, whose halves hold each
/// colour exactly.
const FORMAT: PixelFormat = PixelFormat::RGBA16Float;

/// A parent's width and height at level 0.
const SIDE: u32 = 8;

/// A parent's levels: 8, 4 and 2 texels a side.
const LEVELS: u32 = 3;

/// The threads of a sampling dispatch's threadgroup; its grid is the points padded to a multiple.
const THREADS: u32 = 64;

/// A sample's bytes: four `float`s.
const SAMPLE: u32 = 16;

/// A 2D array parent's slices: more than a cube's six, so a cube of six of them starts past its
/// first and ends before its last.
const ARRAY_SLICES: u32 = 8;

/// One case: the parent's type and slices, and the view's type and ranges.
#[derive(Clone, Copy, Debug)]
struct Pair {
    case: &'static str,
    parent: TextureType,
    /// The parent's slices as Metal counts a view's `sliceRange`: six for a cube, one for a 2D
    /// texture.
    parent_slices: u32,
    view: TextureType,
    levels: (u32, u32),
    slices: (u32, u32),
}

const fn pair(
    case: &'static str,
    parent: TextureType,
    view: TextureType,
    levels: (u32, u32),
    slices: (u32, u32),
) -> Pair {
    let parent_slices = match parent {
        TextureType::Cube => 6,
        TextureType::Type2DArray => ARRAY_SLICES,
        _ => 1,
    };
    Pair {
        case,
        parent,
        parent_slices,
        view,
        levels,
        slices,
    }
}

/// The cases, as the module's table states them: each view's levels and slices as
/// `(first, count)`.
const PAIRS: [Pair; 5] = [
    pair(
        "texture-view-casts-cube-as-2d-array-whole",
        Cube,
        Type2DArray,
        (0, 3),
        (0, 6),
    ),
    pair(
        "texture-view-casts-cube-as-2d-array-faces",
        Cube,
        Type2DArray,
        (1, 2),
        (2, 3),
    ),
    pair(
        "texture-view-casts-cube-as-2d",
        Cube,
        Type2D,
        (1, 1),
        (4, 1),
    ),
    pair(
        "texture-view-casts-2d-as-2d-array",
        Type2D,
        Type2DArray,
        (1, 2),
        (0, 1),
    ),
    pair(
        "texture-view-casts-2d-array-as-cube",
        Type2DArray,
        Cube,
        (1, 2),
        (1, 6),
    ),
];

/// A texture's shape: its levels, and its slices as Metal counts a view's `sliceRange`.
#[derive(Clone, Copy, Debug)]
struct Shape {
    levels: u32,
    slices: u32,
}

impl Shape {
    /// The extent of `level`, square.
    const fn side(&self, level: u32, first_level: u32) -> u32 {
        let side = SIDE >> (first_level + level);
        if side == 0 {
            1
        } else {
            side
        }
    }
}

/// Which of the two passes of a case: what is rendered into, and what is sampled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Way {
    /// Rendered into the view, the parent sampled.
    IntoTheView,
    /// Rendered into the parent, the view sampled.
    IntoTheParent,
}

impl Way {
    const BOTH: [Way; 2] = [Way::IntoTheView, Way::IntoTheParent];

    const fn site(self) -> &'static str {
        match self {
            Way::IntoTheView => "into-the-view",
            Way::IntoTheParent => "into-the-parent",
        }
    }

    /// The blue channel each way's colours carry, so a sample of what the other way left reads
    /// as a difference.
    const fn blue(self) -> f64 {
        match self {
            Way::IntoTheView => 0.25,
            Way::IntoTheParent => 0.75,
        }
    }
}

/// The colour a way clears level `level` and slice `slice` of what it renders into to, in that
/// texture's own levels and slices: each channel a half exactly.
fn colour(way: Way, level: u32, slice: u32) -> [f64; 4] {
    [
        f64::from(slice + 1) / 16.0,
        f64::from(level + 1) / 8.0,
        way.blue(),
        1.0,
    ]
}

/// The sampler of the fixture that samples a texture of `texture_type`.
fn kernel(texture_type: TextureType) -> &'static str {
    match texture_type {
        Type2D => "tvc_2d",
        Type2DArray => "tvc_2d_array",
        _ => "tvc_cube",
    }
}

impl Pair {
    fn parent_shape(&self) -> Shape {
        Shape {
            levels: LEVELS,
            slices: self.parent_slices,
        }
    }

    fn view_shape(&self) -> Shape {
        Shape {
            levels: self.levels.1,
            slices: self.slices.1,
        }
    }

    /// The parent's descriptor: `arrayLength` is a 2D array's slices, and 1 for a cube, whose six
    /// faces it does not count.
    fn parent_request(&self) -> TextureRequest<'static> {
        TextureRequest {
            format: FORMAT.as_raw(),
            texture_type: self.parent,
            size: [SIDE, SIDE, 1],
            levels: LEVELS,
            slices: match self.parent {
                Type2DArray => self.parent_slices,
                _ => 1,
            },
            usage: TextureUsage::RENDER_TARGET.union(TextureUsage::SHADER_READ),
            private: true,
            gpu_optimized: true,
            swizzle: TextureSwizzleChannels::DEFAULT,
            over: None,
            uploads: &[],
        }
    }

    fn view_request(&self) -> ViewRequest {
        ViewRequest {
            call: ViewCall::LevelsSlices,
            format: FORMAT,
            texture_type: self.view,
            first_level: u64::from(self.levels.0),
            level_count: u64::from(self.levels.1),
            first_slice: u64::from(self.slices.0),
            slice_count: u64::from(self.slices.1),
            swizzle: TextureSwizzleChannels::DEFAULT,
        }
    }

    /// What `way` renders into and samples: the rendered texture's shape and first level, and
    /// the sampled texture's shape, first level, and the levels and slices of it sampled, each
    /// with the level and slice of the rendered texture that lies there.
    fn plan(&self, way: Way) -> Plan {
        let parent = self.parent_shape();
        let view = self.view_shape();
        let (first_level, first_slice) = (self.levels.0, self.slices.0);
        match way {
            Way::IntoTheView => Plan {
                rendered: view,
                sampled: parent,
                sampled_first: 0,
                spots: (0..view.levels)
                    .flat_map(|level| {
                        (0..view.slices).map(move |slice| Spot {
                            sampled: [first_level + level, first_slice + slice],
                            rendered: [level, slice],
                        })
                    })
                    .collect(),
            },
            Way::IntoTheParent => Plan {
                rendered: parent,
                sampled: view,
                sampled_first: first_level,
                spots: (0..view.levels)
                    .flat_map(|level| {
                        (0..view.slices).map(move |slice| Spot {
                            sampled: [level, slice],
                            rendered: [first_level + level, first_slice + slice],
                        })
                    })
                    .collect(),
            },
        }
    }
}

/// One level and slice sampled, and the level and slice of the rendered texture that lie there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Spot {
    sampled: [u32; 2],
    rendered: [u32; 2],
}

/// What one way of a case renders and samples ([`Pair::plan`]).
#[derive(Clone, Debug)]
struct Plan {
    rendered: Shape,
    sampled: Shape,
    /// The sampled texture's first level in the parent's: a view's level 0 is the parent's level
    /// under it, and its extent is that level's.
    sampled_first: u32,
    spots: Vec<Spot>,
}

impl Plan {
    /// The points the dispatch samples, `(level, slice, x, y)` in the sampled texture's own
    /// levels and slices, spot by spot and row by row, padded to a multiple of [`THREADS`] with
    /// the last.
    fn points(&self) -> Vec<[u32; 4]> {
        let mut points = Vec::new();
        for spot in &self.spots {
            let [level, slice] = spot.sampled;
            let side = self.sampled.side(level, self.sampled_first);
            for y in 0..side {
                for x in 0..side {
                    points.push([level, slice, x, y]);
                }
            }
        }
        let last = *points.last().expect("every plan samples a texel");
        while !points.len().is_multiple_of(THREADS as usize) {
            points.push(last);
        }
        points
    }

    /// The texels each spot samples, in the order [`Plan::points`] lists them.
    fn texels(&self, spot: &Spot) -> u32 {
        let side = self.sampled.side(spot.sampled[0], self.sampled_first);
        side * side
    }

    /// The passes: one a level and slice of the rendered texture, clearing it to its colour.
    fn passes(&self, way: Way, rendered: Handle) -> Vec<RenderWork> {
        let mut work = Vec::new();
        for level in 0..self.rendered.levels {
            for slice in 0..self.rendered.slices {
                let color = PassColor {
                    place: AttachmentPlace {
                        level,
                        slice,
                        ..AttachmentPlace::BASE
                    },
                    ..PassColor::cleared(rendered, colour(way, level, slice))
                };
                work.push(RenderWork::Pass(RenderPass::clearing(vec![color])));
            }
        }
        work
    }
}

/// The points as the buffer the sampler reads: four `uint`s each.
fn point_bytes(points: &[[u32; 4]]) -> Vec<u8> {
    points
        .iter()
        .flatten()
        .flat_map(|word| word.to_le_bytes())
        .collect()
}

/// A case: one pair.
pub struct TextureViewCast {
    pair: &'static Pair,
}

/// Every case of the family.
pub fn all() -> Vec<Box<dyn Case>> {
    PAIRS
        .iter()
        .map(|pair| Box::new(TextureViewCast { pair }) as _)
        .collect()
}

impl Case for TextureViewCast {
    fn name(&self) -> &'static str {
        self.pair.case
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // The samples land in a `Shared` buffer, and the points are read from one, which is what
        // [`crate::replay::shared_memory_notice`] is for; it loosens the notices site alone, and
        // every sample still compares exactly.
        vec![crate::replay::shared_memory_notice()]
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        let fixture = Fixture::new(FIXTURE);
        let entry = [kernel(self.pair.parent), kernel(self.pair.view)].join("+");
        let mut shaders = Vec::new();
        for extension in [RUNTIME_TEXT, METAL_TEXT] {
            shaders.push(ShaderRef {
                fixture: format!("{FIXTURE}{extension}"),
                entry: entry.clone(),
                source_sha256: crate::sha256_hex(fixture.text(extension)?.as_bytes()),
            });
        }
        Ok(shaders)
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let fixture = Fixture::new(FIXTURE);
        let pair = self.pair;
        let mut inputs = vec![
            Input {
                name: "air".to_owned(),
                digest: Digest::sha256(fixture.text(RUNTIME_TEXT)?.as_bytes()),
            },
            Input {
                name: "air-typed".to_owned(),
                digest: Digest::sha256(fixture.text(METAL_TEXT)?.as_bytes()),
            },
            Input::bytes("texture:parent", &pair.parent_request().words()),
            Input::bytes("view", &pair.view_request().words()),
        ];
        for way in Way::BOTH {
            let plan = pair.plan(way);
            let mut clears = Vec::new();
            for level in 0..plan.rendered.levels {
                for slice in 0..plan.rendered.slices {
                    for channel in colour(way, level, slice) {
                        clears.extend(channel.to_le_bytes());
                    }
                }
            }
            inputs.push(Input::bytes(format!("{}:clears", way.site()), &clears));
            inputs.push(Input::bytes(
                format!("{}:points", way.site()),
                &point_bytes(&plan.points()),
            ));
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let pair = self.pair;
        let loaded = Fixture::new(FIXTURE).loaded().map_err(|why| Abort {
            at: "library".to_owned(),
            failure: Failure::named("FixtureNotRead", why),
        })?;
        let library = made(sites, "library", side.library(&loaded.source()))?;
        let parent_sampler = made(
            sites,
            "pipeline:parent",
            side.compute_pipeline(library, kernel(pair.parent)),
        )?;
        let view_sampler = made(
            sites,
            "pipeline:view",
            side.compute_pipeline(library, kernel(pair.view)),
        )?;
        let parent = made(sites, "parent", side.texture(&pair.parent_request()))?;
        let view = made(
            sites,
            "view",
            side.texture_view(parent, &pair.view_request()),
        )?;
        let facts = call(
            sites,
            "view-facts",
            side.texture_view_facts(view, Some(parent)),
            |_| Value::Absent,
        )?;
        // The view's own descriptor, which the casts are about; its parent's getters are another
        // family's (`texture-view-parents`).
        sites.push(
            Node::branch("view-descriptor", Vec::new())
                .with(Node::leaf(
                    "texture_type",
                    Observed::value(Value::Uint(facts.texture_type)),
                ))
                .with(Node::leaf(
                    "pixel_format",
                    Observed::value(Value::Uint(facts.pixel_format)),
                ))
                .with(Node::leaf(
                    "mipmap_level_count",
                    Observed::value(Value::Uint(facts.mipmap_level_count)),
                ))
                .with(Node::leaf(
                    "array_length",
                    Observed::value(Value::Uint(facts.array_length)),
                )),
        );
        for way in Way::BOTH {
            let plan = pair.plan(way);
            let (rendered, sampled, sampler) = match way {
                Way::IntoTheView => (view, parent, parent_sampler),
                Way::IntoTheParent => (parent, view, view_sampler),
            };
            let points = plan.points();
            let at = |what: &str| format!("{}:{what}", way.site());
            let points_buffer = made(sites, &at("points"), side.buffer(&point_bytes(&points)))?;
            let length = points.len() * SAMPLE as usize;
            let out = made(sites, &at("samples"), side.zeroed_buffer(length))?;
            let passes = plan.passes(way, rendered);
            let outcome = call(sites, &at("render"), side.render(&passes), |outcome| {
                outcome.status.clone()
            })?;
            if let Some(failure) = outcome.refused.iter().flatten().next() {
                return Err(Abort {
                    at: at("render"),
                    failure: failure.clone(),
                });
            }
            let dispatch = RenderWork::Dispatch(Dispatch {
                pipeline: sampler,
                buffers: vec![(points_buffer, 0), (out, 0)],
                textures: vec![sampled],
                grid: [points.len() as u32, 1, 1],
                threads: [THREADS, 1, 1],
            });
            let outcome = call(sites, &at("sample"), side.render(&[dispatch]), |outcome| {
                outcome.status.clone()
            })?;
            if let Some(failure) = outcome.refused.iter().flatten().next() {
                return Err(Abort {
                    at: at("sample"),
                    failure: failure.clone(),
                });
            }
            let bytes = call(sites, &at("read"), side.read_buffer(out, 0, length), |_| {
                Value::Absent
            })?;
            let mut node = Node::branch(way.site(), Vec::new());
            let mut offset = 0u32;
            for spot in &plan.spots {
                let texels = plan.texels(spot);
                let side_texels = plan.sampled.side(spot.sampled[0], plan.sampled_first);
                let span = texels * SAMPLE;
                let from = offset as usize;
                let held = bytes[from..from + span as usize].to_vec();
                let [level, slice] = spot.sampled;
                node = node.with(Node::leaf(
                    format!("level-{level}-slice-{slice}"),
                    Observed::bytes(held, sample_layout(offset, side_texels)),
                ));
                offset += span;
            }
            sites.push(node);
        }
        Ok(())
    }
}

/// The layout a spot's samples are read under: one row of the layout a row of texels, one
/// element a channel.
fn sample_layout(offset: u32, side: u32) -> ByteLayout {
    let row_pitch = SAMPLE * side;
    ByteLayout {
        region: Region::Buffer {
            offset: u64::from(offset),
            length: u64::from(row_pitch * side),
        },
        element: Element::F32,
        row_offset: 0,
        row_elements: 4 * side,
        row_pitch,
        rows: side,
        image_pitch: row_pitch * side,
        images: 1,
        gaps: Gaps::Compared,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::texture_views::ParentShape;

    /// The parent's shape as the view guard reads it.
    fn parent_view_shape(pair: &Pair) -> ParentShape {
        ParentShape {
            texture_type: pair.parent,
            format: FORMAT,
            levels: u64::from(LEVELS),
            slices: u64::from(pair.parent_slices),
            is_a_view: false,
        }
    }

    /// No view of the family is one Metal's validation stops, which the request's own guard
    /// answers for with no device: the proof that a row added by mistake cannot end Metal's
    /// process.
    #[test]
    fn no_view_of_the_family_is_one_metal_s_validation_stops() {
        for pair in &PAIRS {
            assert_eq!(
                pair.view_request().is_well_formed(&parent_view_shape(pair)),
                None,
                "{}",
                pair.case
            );
        }
    }

    /// Every view lies inside its parent at a level or slice past the parent's first somewhere,
    /// so a side that ignored the ranges reads otherwise; the streams' form, every level and all
    /// six faces of a cube as a 2D array, is one of them; and every cast metal2vulkan-runtime builds of
    /// another type than the parent's but one slice of a 2D array as a 2D texture, which the
    /// suite's other view families make, is a case, a square 2D array's cube among them.
    #[test]
    fn the_table_holds_the_streams_form_and_every_cast_built() {
        assert!(PAIRS
            .iter()
            .any(|pair| pair.levels.0 > 0 || pair.slices.0 > 0));
        assert!(PAIRS.iter().any(|pair| pair.parent == Cube
            && pair.view == Type2DArray
            && pair.levels == (0, LEVELS)
            && pair.slices == (0, 6)));
        for (parent, view) in [
            (Cube, Type2D),
            (Cube, Type2DArray),
            (Type2D, Type2DArray),
            (Type2DArray, Cube),
        ] {
            assert!(
                PAIRS
                    .iter()
                    .any(|pair| pair.parent == parent && pair.view == view),
                "{parent:?} as {view:?}"
            );
        }
        let mut names: Vec<&str> = PAIRS.iter().map(|pair| pair.case).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), PAIRS.len());
    }

    /// Metal's recorded samples are, at every texel, the colour the way cleared the level and
    /// slice that lies there: the views and parents share their texels as the module states, and
    /// a side that answers otherwise differs from Metal rather than from a guess.
    #[test]
    fn every_recorded_sample_is_the_colour_its_way_cleared() {
        for pair in &PAIRS {
            let Ok(recording) = crate::recording::read(pair.case) else {
                panic!("{}: no recording", pair.case);
            };
            let answer = recording.answers.first().expect("an answer");
            assert_eq!(answer.outcomes.len(), 1, "{}", pair.case);
            let root = &answer.outcomes[0].transcript.root;
            let case = root.child("case").unwrap_or(root);
            for way in Way::BOTH {
                let plan = pair.plan(way);
                let node = case
                    .child(way.site())
                    .unwrap_or_else(|| panic!("{}: no {}", pair.case, way.site()));
                for spot in &plan.spots {
                    let [level, slice] = spot.sampled;
                    let at = format!("level-{level}-slice-{slice}");
                    let leaf = node
                        .child(&at)
                        .unwrap_or_else(|| panic!("{}: no {at}", pair.case));
                    let Value::Bytes(bytes) = &leaf.observed.value else {
                        panic!("{}: {at} holds no bytes", pair.case);
                    };
                    let want = colour(way, spot.rendered[0], spot.rendered[1]).map(|c| c as f32);
                    assert_eq!(bytes.len() as u32, plan.texels(spot) * SAMPLE, "{at}");
                    for sample in bytes.chunks(SAMPLE as usize) {
                        let got: Vec<f32> = sample
                            .chunks(4)
                            .map(|word| f32::from_le_bytes(word.try_into().expect("a word")))
                            .collect();
                        assert_eq!(got, want, "{}: {}: {at}", pair.case, way.site());
                    }
                }
            }
        }
    }

    /// Each way samples every texel of every level and slice it names once, and the colours of
    /// two levels or two slices of what it renders differ, so a sample of the wrong one reads as
    /// a difference.
    #[test]
    fn every_spot_is_sampled_once_and_its_colour_is_its_own() {
        for pair in &PAIRS {
            for way in Way::BOTH {
                let plan = pair.plan(way);
                let texels: u32 = plan.spots.iter().map(|spot| plan.texels(spot)).sum();
                let points = plan.points();
                assert!(points.len() >= texels as usize, "{}", pair.case);
                assert_eq!(points.len() % THREADS as usize, 0, "{}", pair.case);
                let mut colours: Vec<[u64; 4]> = plan
                    .spots
                    .iter()
                    .map(|spot| colour(way, spot.rendered[0], spot.rendered[1]).map(f64::to_bits))
                    .collect();
                let spots = colours.len();
                colours.sort_unstable();
                colours.dedup();
                assert_eq!(colours.len(), spots, "{}", pair.case);
                for spot in &plan.spots {
                    let [level, slice] = spot.rendered;
                    assert!(level < plan.rendered.levels && slice < plan.rendered.slices);
                    // A half holds every multiple of 1/16 in [0, 1] exactly.
                    for channel in colour(way, level, slice) {
                        let sixteenths = channel * 16.0;
                        assert!(
                            (0.0..=1.0).contains(&channel) && sixteenths.fract() == 0.0,
                            "{}: {channel} is a half exactly",
                            pair.case
                        );
                    }
                }
            }
        }
    }
}
