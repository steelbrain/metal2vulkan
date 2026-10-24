//! What a render pass stores into the IOSurface plane behind a framebuffer-only **rotated** texture.
//!
//! # Why this is recorded rather than carried
//!
//! The runtime lane `rotated_textures` checks every plane byte against a *rule* — where a pass's
//! texel `(x, y)` lands at each rotation, and what a load leaves at the texels a draw does not
//! reach. That rule was measured on an Apple M4 under macOS 27.0.1, a chip this tree no longer has,
//! which makes it the weakest kind of row: it can never be checked against the oracle again.
//!
//! So the rule is **re-measured here**, on an Apple M3 Max under macOS 27.0.1 (26A434), and the
//! cases record Metal's own plane bytes rather than the rule's prediction. The re-measurement
//! reproduced the M4's rule exactly, all five rotations, every texel of an 8x4 frame, store and
//! load alike (`probes/rotated_surface_planes.m`) — so the lane's numbers were right, and they are
//! now a fact on the oracle instead of a number from a machine that is gone.
//!
//! # What the Metal leg needs, and why it is private
//!
//! `framebufferOnly` and `rotation` are Apple SPI on `MTLTextureDescriptor`
//! (`setFramebufferOnly:`, `setRotation:`), and Metal makes such a texture **only over an
//! IOSurface** — `framebufferOnly` over anything else makes it assert. Private Metal is in scope
//! for this project, so the Metal side reaches both setters by selector after
//! `respondsToSelector:`, over a real IOSurface, and the runtime's leg runs over the buffer that
//! stands for one. Both were verified answered on this chip; the descriptor reads the rotation
//! back.
//!
//! # The stride rule, which is the safety story
//!
//! **Metal asserts, ending the process, on an IOSurface texture whose `bytesPerRow` is below the
//! TEXTURE's own row** — not the surface's, whatever the rotation. That was measured the hard way:
//! the first run of the probe died on
//!
//! ```text
//! _mtlValidateStrideTextureParameters: IOSurface texture: bytesPerRow (16) must be greater or
//! equal to (32) bytes
//! ```
//!
//! for a 4x8 surface at its own tight stride behind an 8x4 texture — which is exactly the plane the
//! lane this ports uses for a quarter turn. So **every plane here carries a stride of at least the
//! frame's own row**, [`crate::surface::PlaneSpec::unsafe_stride`] refuses the rest, and both sides
//! check it at the call rather than trusting a case body. Nothing in this module can end Metal's
//! process, fault the GPU or hang it: one single-sampled pass a case, one triangle, no loop, and
//! every byte written inside the surface's own allocation.
//!
//! That rule is also a **finding**, recorded in this stream's report: the lane passes with a plane
//! stride Metal would assert on, so the runtime accepts a descriptor plain Metal refuses.
//!
//! # The shape
//!
//! Two questions a rotation, over two frames:
//!
//! - **store**: a pass that clears the frame and draws every texel of a source texture
//!   whose texels name themselves, so each plane byte names the pass texel Metal stored there. That
//!   is `stored_at` measured directly.
//! - **load**: the plane's texels each named with their own stored position, then a pass
//!   that loads and draws only the top-left 8x4 pass texels. What the undrawn texels keep is the
//!   answer, and at rotation 4 it is the one place our runtime is known to differ from Metal.
//!
//! The frames are 64x16, whose turned plane is 16x64 and so spans **two 32x32 plane tiles**, and
//! 32x32, which is one tile. The two-tile frame is deliberate: measured here, at rotation 4 Metal
//! mirrors the loaded texels left to right in the tile the draw reached (480 of 480 undrawn texels)
//! and **keeps every texel of the tile it did not reach** (512 of 512), while the runtime mirrors
//! every tile and announces `RotatedLoadTiles`. A one-tile frame cannot tell those apart, which is
//! why the lane's own 32x32 test never did.

use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Case, Input, MetalSource, Side};
use crate::recording::ShaderRef;
use crate::render::{
    ColorState, PassColor, RenderPass, RenderPipelineSpec, RenderStep, RenderWork, TargetSpec,
};
use crate::surface::{plane_extent, PlaneSpec};
use crate::transcript::{Node, Observed, Sites, Value};
use metal2vulkan::metal::PixelFormat;
use metal2vulkan_runtime::{Degradation, LoadAction};

use super::extended_range_targets::{fixture, fixture_inputs, not_read, program, shader_refs};

/// The format a rotated frame is, as the consumer's surface is.
const PF: PixelFormat = PixelFormat::BGRA8Unorm;
/// Four bytes a texel.
const TEXEL: u32 = 4;
/// The byte every plane byte holds before a case writes anything, so a byte no pass touched is
/// visibly untouched rather than zero.
const PAD: u8 = 0xab;

/// The owned fixture whose rectangle and texture read the passes use.
const RENDERS: &str = "tob_renders";
/// The vertex stage: a rectangle whose corners come from vertex buffer 0.
const RECT: &str = "tob_rect";
/// The fragment stage: reads, at its own position, a texture whose texels name themselves, so a
/// drawn texel names the pass position it was drawn at.
const READ: &str = "tob_read";

/// The pass texels a partial draw covers, the lane's own corner.
const DRAWN: [u32; 2] = [8, 4];

/// What a case asks of one rotation.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Ask {
    /// A pass that clears the frame and draws every texel: where Metal stores each pass texel.
    Store,
    /// The plane's texels named with their own stored position, then a pass that loads and draws
    /// the top-left corner: what the undrawn texels keep.
    Load,
}

/// A case: one rotation of one frame, asked one of the two questions.
pub struct RotatedPlane {
    /// The case's name, a literal, which is also its recording's file name.
    name: &'static str,
    ask: Ask,
    rotation: u32,
    frame: [u32; 2],
}

/// Every case: both questions, five rotations, two frames.
///
/// The names are written out rather than built, so a recording's file name is something a reader
/// can grep for in this file.
pub fn all() -> Vec<Box<dyn Case>> {
    const ROWS: [(&str, Ask, u32, [u32; 2]); 20] = [
        ("rotated-planes-store-0-64x16", Ask::Store, 0, [64, 16]),
        ("rotated-planes-store-1-64x16", Ask::Store, 1, [64, 16]),
        ("rotated-planes-store-2-64x16", Ask::Store, 2, [64, 16]),
        ("rotated-planes-store-3-64x16", Ask::Store, 3, [64, 16]),
        ("rotated-planes-store-4-64x16", Ask::Store, 4, [64, 16]),
        ("rotated-planes-store-0-32x32", Ask::Store, 0, [32, 32]),
        ("rotated-planes-store-1-32x32", Ask::Store, 1, [32, 32]),
        ("rotated-planes-store-2-32x32", Ask::Store, 2, [32, 32]),
        ("rotated-planes-store-3-32x32", Ask::Store, 3, [32, 32]),
        ("rotated-planes-store-4-32x32", Ask::Store, 4, [32, 32]),
        ("rotated-planes-load-0-64x16", Ask::Load, 0, [64, 16]),
        ("rotated-planes-load-1-64x16", Ask::Load, 1, [64, 16]),
        ("rotated-planes-load-2-64x16", Ask::Load, 2, [64, 16]),
        ("rotated-planes-load-3-64x16", Ask::Load, 3, [64, 16]),
        ("rotated-planes-load-4-64x16", Ask::Load, 4, [64, 16]),
        ("rotated-planes-load-0-32x32", Ask::Load, 0, [32, 32]),
        ("rotated-planes-load-1-32x32", Ask::Load, 1, [32, 32]),
        ("rotated-planes-load-2-32x32", Ask::Load, 2, [32, 32]),
        ("rotated-planes-load-3-32x32", Ask::Load, 3, [32, 32]),
        ("rotated-planes-load-4-32x32", Ask::Load, 4, [32, 32]),
    ];
    ROWS.iter()
        .map(|(name, ask, rotation, frame)| {
            Box::new(RotatedPlane {
                name,
                ask: *ask,
                rotation: *rotation,
                frame: *frame,
            }) as Box<dyn Case>
        })
        .collect()
}

impl RotatedPlane {
    /// The plane the frame's store lands in: the quarter turn's extent, at a stride of the frame's
    /// own row plus 16 bytes.
    ///
    /// The padding is the point of the stride: a case asserts that the bytes between the rows
    /// stayed as they were, which needs rows that are wider than their texels. **The stride is at
    /// least the frame's own row**, which is the rule Metal asserts on
    /// ([the module's header](self)); for a quarter turn that is wider than the plane's texels,
    /// which is exactly the case the lane this ports got wrong.
    fn plane(&self) -> PlaneSpec {
        let plane = plane_extent(self.rotation, self.frame);
        let stride = (plane[0].max(self.frame[0])) * TEXEL + 16;
        PlaneSpec::new(self.rotation, plane, stride, TEXEL, PAD)
    }

    /// The target: a framebuffer-only rotated texture over that plane.
    fn target(&self) -> TargetSpec {
        TargetSpec::read(PF, self.frame).over_plane(self.plane())
    }

    /// The source texture's texels: texel `(x, y)` holds the bytes `x`, `y`, `0x55`, `0xff`, so a
    /// drawn plane byte names the pass texel it came from.
    fn naming(&self) -> Vec<u8> {
        let [width, height] = self.frame;
        let mut bytes = Vec::with_capacity((width * height * TEXEL) as usize);
        for y in 0..height {
            for x in 0..width {
                bytes.extend([x as u8, y as u8, 0x55, 0xff]);
            }
        }
        bytes
    }

    /// The plane's bytes a [`Ask::Load`] case writes before its pass: every plane texel naming its
    /// own stored position (`sx`, `sy`, `0x33`, `0x77`), the bytes between the rows left as the
    /// pad.
    ///
    /// The two marker bytes differ from the source texture's (`0x55`, `0xff`) so a plane byte says
    /// which it is: a texel the pass drew, or one it loaded and kept.
    fn named_plane(&self) -> Vec<u8> {
        let plane = self.plane();
        let [width, height] = plane.plane;
        let mut bytes = vec![PAD; plane.length()];
        for sy in 0..height {
            for sx in 0..width {
                let at = (sy * plane.row_pitch + sx * TEXEL) as usize;
                bytes[at..at + 4].copy_from_slice(&[sx as u8, sy as u8, 0x33, 0x77]);
            }
        }
        bytes
    }

    /// The rectangle's corners in clip space: the whole frame for a store, and the top-left
    /// [`DRAWN`] pass texels for a load.
    fn corners(&self) -> Vec<u8> {
        let [width, height] = self.frame.map(|axis| axis as f32);
        let corners = match self.ask {
            Ask::Store => [-1.0, -1.0, 1.0, 1.0],
            Ask::Load => [
                -1.0,
                1.0 - 2.0 * DRAWN[1] as f32 / height,
                -1.0 + 2.0 * DRAWN[0] as f32 / width,
                1.0,
            ],
        };
        corners
            .iter()
            .flat_map(|corner: &f32| corner.to_le_bytes())
            .collect()
    }

    /// The pass's load action: a store case clears the frame, a load case loads it.
    fn load(&self) -> LoadAction {
        match self.ask {
            Ask::Store => LoadAction::Clear,
            Ask::Load => LoadAction::Load,
        }
    }

    /// The attachment the pipeline draws into.
    fn colors(&self) -> [Option<ColorState>; 1] {
        [Some(ColorState::new(PF))]
    }

    /// The 32-row bands of the plane, which is what a 32x32 tile is for these frames.
    ///
    /// Both frames here turn into a plane at most 32 texels wide (16x64 and 32x32), so a tile
    /// column is the whole row and a tile is a band of 32 rows. The case says that rather than
    /// pretending to a general 2D tiling it does not exercise.
    fn bands(&self) -> Vec<(u32, u32)> {
        let plane = self.plane();
        let mut bands = Vec::new();
        let mut first = 0;
        while first < plane.plane[1] {
            let rows = 32.min(plane.plane[1] - first);
            bands.push((first, rows));
            first += rows;
        }
        bands
    }

    /// Whether the draw reaches any texel of the band starting at row `first`.
    ///
    /// Derived from where the draw's own pass texels are stored
    /// ([`crate::surface::stored_at`]), so it is a structural answer rather than a band index
    /// written down for one rotation.
    fn band_is_reached(&self, first: u32, rows: u32) -> bool {
        if self.ask == Ask::Store {
            return true;
        }
        (0..DRAWN[1]).any(|y| {
            (0..DRAWN[0]).any(|x| {
                let [_, sy] = crate::surface::stored_at(self.rotation, self.frame, [x, y]);
                sy >= first && sy < first + rows
            })
        })
    }

    /// The layout one band of the plane is read under: its own rows at the plane's stride, so a
    /// diff names the plane texel that moved **and the gaps between the rows are compared** -- a
    /// case of this family asserts the bytes between the rows stayed as they were, which is half of
    /// what it is asking.
    ///
    /// One node a band rather than one for the whole plane, because the one place the runtime
    /// differs from Metal is **a whole tile**: Metal mirrors a rotation-4 load only in the tile the
    /// draw reached and keeps the tile it did not. A single `plane` node would force the allowance
    /// for that to loosen every byte of the plane, including the tile that must still match exactly.
    /// Per band, the allowance loosens the unreached tile alone.
    fn band_layout(&self, first: u32, rows: u32) -> crate::transcript::ByteLayout {
        use crate::transcript::{ByteLayout, Element, Gaps, Region};
        let plane = self.plane();
        let offset = u64::from(first) * u64::from(plane.row_pitch);
        ByteLayout {
            region: Region::Buffer {
                offset,
                length: u64::from(rows) * u64::from(plane.row_pitch),
            },
            element: Element::U8,
            row_offset: 0,
            row_elements: plane.plane[0] * TEXEL,
            row_pitch: plane.row_pitch,
            rows,
            image_pitch: plane.row_pitch * rows,
            images: 1,
            // The rows are wider than their texels, and whether the bytes between them stayed as
            // the pad is part of the answer.
            gaps: Gaps::Compared,
        }
    }
}

impl Case for RotatedPlane {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        // Nothing here carries. `setFramebufferOnly:` and `setRotation:` are Apple SPI, but a
        // private selector is not a reason to carry an answer -- it is a reason to check
        // `respondsToSelector:` first. The rest is public: IOSurface is a public framework,
        // `newTextureWithDescriptor:iosurface:plane:` is public, and the pass is an ordinary one.
        MetalSource::Run
    }

    fn allowances(&self) -> Vec<Allowance> {
        // A load at rotation 4 is the one shape where the runtime differs from Metal, and it says
        // so: Metal mirrors the loaded texels only in each 32x32 tile the draw reached, and the
        // runtime mirrors every tile, announced once as `RotatedLoadTiles`.
        //
        // The allowance is only taken by the frame where the difference can actually appear. A
        // 32x32 frame is one tile, so there is no unreached tile and nothing to loosen -- naming
        // the allowance there would be a claim with no evidence, and the allowance machinery would
        // rightly fail it if the notice did not fire.
        // Every case of this family announces one notice Metal has no counterpart for: the runtime
        // reads and renders a rotated texture through shadows (`ShadowedTexture`). Metal announces
        // nothing, because Metal has no notices at all, so the notice set is loosened for the
        // family -- and only the notice set, so every plane byte still compares exactly.
        // A device whose own conversion of a float written to the plane's normalized attachment
        // is not Metal's converts it in the shader and announces `shader-color-conversion`
        // (lavapipe). Declared for the family: every plane byte still compares exactly.
        // The buffer every case reads its plane back through is `Shared`, which a device with no
        // unified memory keeps in system memory because that is the only memory both sides can
        // reach (`shared-in-system-memory`): the venue's memory model, not a rotation.
        let mut allowances = vec![
            crate::replay::notice_allowance(Degradation::ShadowedTexture.code()),
            crate::replay::notice_allowance(Degradation::ShaderColorConversion.code()),
            crate::replay::shared_memory_notice(),
        ];
        if self.ask == Ask::Load && self.rotation == 4 {
            // The notice is announced wherever rotation 4 loads, whatever the frame, so its code
            // is **declared** here for every such frame: Metal announces none, and a notice our
            // side announces that the case names nowhere is a difference with nothing accounting
            // for it. The declaration loosens no byte. It is not the tolerance below, which stays
            // held to the bands where the difference can appear -- and that is why a 32x32 frame
            // needs this line: it has no unreached band to loosen, and announced the notice all
            // the same.
            allowances.push(crate::replay::notice_allowance(
                Degradation::RotatedLoadTiles.code(),
            ));
            // The one place the runtime is known to differ from Metal, and the allowance is held to
            // exactly where it differs.
            //
            // MEASURED on this M3 Max, from Metal's own recorded plane bytes: a rotation-4 load
            // leaves the loaded texels mirrored left to right in the 32x32 plane tile the draw
            // reached (480 of 480 undrawn texels) and **keeps every texel of the tile it did not
            // reach** (512 of 512). The runtime mirrors every tile and announces
            // `RotatedLoadTiles`.
            //
            // So only the bands the draw does not reach are loosened. The reached band still
            // compares byte for byte, which is where the mirror itself is checked, and a frame of
            // one band (32x32) has nothing to loosen at all -- naming an allowance there would be a
            // claim with no evidence.
            allowances.extend(
                self.bands()
                    .iter()
                    .filter(|&&(first, rows)| !self.band_is_reached(first, rows))
                    .map(|&(first, _)| {
                        crate::allowance::Allowance::new(
                            format!("case/plane/rows-{first}"),
                            crate::allowance::Loosening::Unchecked,
                            crate::allowance::Justification::notice(Degradation::RotatedLoadTiles),
                        )
                    }),
            );
        }
        allowances
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shader_refs(RENDERS, &[RECT, READ])
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs("renders", RENDERS)?;
        let colors = self.colors();
        inputs.extend([
            Input::bytes("rotation", &self.rotation.to_le_bytes()),
            Input::bytes(
                "frame",
                &self
                    .frame
                    .iter()
                    .flat_map(|axis| axis.to_le_bytes())
                    .collect::<Vec<u8>>(),
            ),
            // The plane's extent, stride, texel size and pad: a plane that moves is a change to
            // the case's inputs rather than a quiet pass against another plane's answer.
            Input::bytes("plane", &self.plane().words()),
            Input::bytes("target", &self.target().words()),
            Input::bytes(
                "pipeline",
                &RenderPipelineSpec::new(RECT, READ, &colors).words(),
            ),
            Input::bytes("load-action", &self.load().raw().to_le_bytes()),
            Input::bytes("corners", &self.corners()),
            Input::bytes("texture:naming", &self.naming()),
        ]);
        if self.ask == Ask::Load {
            inputs.push(Input::bytes("plane:before", &self.named_plane()));
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let renders = fixture(RENDERS, RECT)
            .loaded()
            .map_err(|why| not_read("library/renders", why))?;
        let library = made(sites, "library/renders", side.library(&renders.source()))?;
        let colors = self.colors();
        let pipeline = made(
            sites,
            "pipeline",
            side.render_pipeline(library, &RenderPipelineSpec::new(RECT, READ, &colors)),
        )?;
        let source = made(
            sites,
            "texture/naming",
            side.texture_2d(PF, self.frame, self.frame[0] * TEXEL, &self.naming()),
        )?;
        let corners = made(sites, "buffer/corners", side.buffer(&self.corners()))?;
        // The target is the case's whole question: a framebuffer-only rotated texture over an
        // IOSurface plane on Metal, over the buffer standing for one on the runtime. A venue that
        // makes no such texture refuses here, which is the honest gap rather than a skip.
        let target = made(sites, "target", side.target(&self.target()))?;

        // A load case writes the plane's texels first, each naming its own stored position, so the
        // texels the draw does not reach say where they were.
        if self.ask == Ask::Load {
            call(
                sites,
                "write-plane",
                side.write_plane(target, &self.named_plane()),
                |_| Value::Absent,
            )?;
        }

        let mut attachment = PassColor::loaded(target);
        attachment.load = self.load();
        attachment.clear = [0.0, 0.0, 0.0, 0.0];
        let work = vec![RenderWork::Pass(
            RenderPass::clearing(vec![attachment]).drawing(vec![
                RenderStep::Pipeline(pipeline),
                RenderStep::VertexBuffer {
                    buffer: corners,
                    offset: 0,
                    index: 0,
                },
                RenderStep::FragmentTexture {
                    texture: source,
                    index: 0,
                },
                // The fixture's rectangle is two triangles, six vertices of one instance.
                RenderStep::Draw {
                    primitive: metal2vulkan::metal::PrimitiveType::Triangle,
                    first: 0,
                    count: 6,
                    instances: 1,
                },
            ]),
        )];
        let outcome = call(sites, "render", side.render(&work), |_| Value::Absent)?;
        program(sites, "program", &outcome, &work);

        // Every byte of the plane, the whole stride of every row, so the bytes between the rows are
        // observed too. It is emitted one 32-row tile band at a time, so the tile rule's allowance
        // can loosen the unreached tile alone.
        let bytes = call(sites, "read-plane", side.read_plane(target), |_| {
            Value::Absent
        })?;
        let plane = self.plane();
        let mut node = Node::branch("plane", Vec::new());
        for (first, rows) in self.bands() {
            let from = (first * plane.row_pitch) as usize;
            let to = from + (rows * plane.row_pitch) as usize;
            let held = bytes.get(from..to).ok_or_else(|| Abort {
                at: "plane".to_owned(),
                failure: crate::case::Failure::named(
                    "PlaneTooShort",
                    format!(
                        "{} bytes for rows {first}..{} at {} a row",
                        bytes.len(),
                        first + rows,
                        plane.row_pitch
                    ),
                ),
            })?;
            node = node.with(Node::leaf(
                format!("rows-{first}"),
                Observed::bytes(held.to_vec(), self.band_layout(first, rows)),
            ));
        }
        sites.push(node);
        Ok(())
    }
}
