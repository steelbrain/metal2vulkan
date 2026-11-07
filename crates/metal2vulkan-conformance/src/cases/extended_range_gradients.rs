//! Gradient fills into `BGRA10_XR` and its `_sRGB` twin, the target a wide-gamut 2D renderer draws
//! its gradients into: what the pass stores, compared code for code.
//!
//! A `CAMetalLayer` that asks for wide gamut hands its client `BGRA10_XR` drawables, and a 2D
//! renderer such as Impeller fills its gradients there by draws that evaluate the gradient in the
//! fragment function from a buffer of color stops, often at four samples resolved into the
//! drawable. Each case here is one pass over a 32 x 32 target cleared to an opaque color, with two
//! draws of the owned fixture `xr_gradients`:
//!
//! - a full-screen triangle of the radial gradient (`xrg_radial_float`, a `float4` output) of four
//!   stops whose colors reach past the format's unit range on both sides, opaque;
//! - then a triangle whose edges cross pixels, of the linear gradient (`xrg_linear_half`, a `half4`
//!   output) of three stops, at alpha 0.6.
//!
//! Blended (`-blended`), both draws blend under premultiplied source-over, the second over what
//! the first left; unblended (`-unblended`), each stores its own color where it covers. At one
//! sample the target is read back; at four samples the resolve texture is, which the pass
//! resolves into at its store and does not keep the samples of, as a wide-gamut client's onscreen
//! pass does.
//!
//! Nothing here can fault a GPU: small targets of the case's own extent, two triangles, a stop loop
//! of at most eight steps, and every fetch inside its buffer.

use crate::allowance::Allowance;
use crate::case::{call, made, Abort, Case, Handle, Input, MetalSource, Side};
use crate::recording::ShaderRef;
use crate::render::{
    Blend, ColorState, PassColor, Plane, ReadAt, RenderPass, RenderPipelineSpec, RenderStep,
    RenderWork, TargetSpec,
};
use crate::transcript::{Node, Observed, Sites, Value};
use metal2vulkan::metal::BlendFactor as F;
use metal2vulkan_runtime::{Degradation, StoreAction};

use super::extended_range_targets::{
    fixture, fixture_inputs, not_read, program, shader_refs, texel_layout, Format, FORMATS,
};

/// The fixture and its entry points.
const GRADIENTS: &str = "xr_gradients";
const VERTEX: &str = "xrg_vertex";
const LINEAR_HALF: &str = "xrg_linear_half";
const RADIAL_FLOAT: &str = "xrg_radial_float";

/// The target's side.
const SIDE: u32 = 32;

/// The clear color: an opaque dark slate.
const CLEAR: [f64; 4] = [0.06, 0.09, 0.16, 1.0];

/// Premultiplied source-over, the client's default blend.
const OVER: Blend = Blend::added(F::One, F::OneMinusSourceAlpha);

/// The full-screen triangle's corners, in clip space.
const FULL: [[f32; 2]; 3] = [[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]];

/// The second draw's triangle, whose edges cross pixels at no symmetry of the grid.
const PART: [[f32; 2]; 3] = [[-0.81, -0.93], [0.96, -0.29], [-0.17, 0.87]];

/// One gradient: its two points (a linear one's from and to, a radial one's center and its radius
/// in `to.x`), its alpha, and its stops (a color, premultiplied by nothing, and a stop).
struct Gradient {
    from: [f32; 2],
    to: [f32; 2],
    alpha: f32,
    stops: &'static [([f32; 4], f32)],
}

/// The first draw's radial gradient: white past the unit range in red, slate, then two dark stops
/// below it in red, opaque.
const RADIAL: Gradient = Gradient {
    from: [0.42, 0.38],
    to: [0.71, 0.0],
    alpha: 1.0,
    stops: &[
        ([1.25, 1.0, 0.9, 1.0], 0.0),
        ([0.39, 0.45, 0.55, 1.0], 0.3),
        ([0.12, 0.16, 0.23, 0.9], 0.75),
        ([-0.05, 0.09, 0.16, 0.8], 1.0),
    ],
};

/// The second draw's linear gradient, translucent.
const LINEAR: Gradient = Gradient {
    from: [0.1, 0.2],
    to: [0.9, 0.7],
    alpha: 0.6,
    stops: &[
        ([0.2, 0.9, -0.1, 1.0], 0.0),
        ([1.1, 0.5, 0.2, 0.7], 0.5),
        ([0.0, 0.3, 1.2, 0.5], 1.0),
    ],
};

impl Gradient {
    /// The stops buffer: each a `float4` color, a `float` stop and 12 bytes of padding
    /// (`XrgStop`'s 32-byte stride).
    fn stop_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.stops.len() * 32);
        for (color, stop) in self.stops {
            for c in color {
                out.extend(c.to_le_bytes());
            }
            out.extend(stop.to_le_bytes());
            out.extend([0u8; 12]);
        }
        out
    }

    /// The info buffer (`XrgInfo`): from, to, the target's extent, the alpha and the stop count.
    fn info_bytes(&self) -> Vec<u8> {
        let side = SIDE as f32;
        let floats = [
            self.from[0],
            self.from[1],
            self.to[0],
            self.to[1],
            side,
            side,
            self.alpha,
        ];
        let mut out: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        out.extend((self.stops.len() as u32).to_le_bytes());
        out
    }
}

/// A triangle's corners as the vertex function reads them.
fn corner_bytes(corners: &[[f32; 2]; 3]) -> Vec<u8> {
    corners
        .iter()
        .flatten()
        .flat_map(|f| f.to_le_bytes())
        .collect()
}

/// One case: two gradient draws into one format at one sample count, blended or not.
pub struct GradientDraw {
    name: &'static str,
    format: Format,
    blended: bool,
    samples: u32,
}

/// Every case: each format, blended and unblended, at one sample and four.
pub fn draws() -> Vec<GradientDraw> {
    const NAMES: [[&str; 4]; 2] = [
        [
            "xr-gradient-bgra10xr-blended-x1",
            "xr-gradient-bgra10xr-blended-x4",
            "xr-gradient-bgra10xr-unblended-x1",
            "xr-gradient-bgra10xr-unblended-x4",
        ],
        [
            "xr-gradient-bgra10xr-srgb-blended-x1",
            "xr-gradient-bgra10xr-srgb-blended-x4",
            "xr-gradient-bgra10xr-srgb-unblended-x1",
            "xr-gradient-bgra10xr-srgb-unblended-x4",
        ],
    ];
    let mut cases = Vec::new();
    for (format, names) in [FORMATS[0], FORMATS[1]].into_iter().zip(NAMES) {
        let mut names = names.into_iter();
        for blended in [true, false] {
            for samples in [1, 4] {
                cases.push(GradientDraw {
                    name: names.next().expect("a name"),
                    format,
                    blended,
                    samples,
                });
            }
        }
    }
    cases
}

impl GradientDraw {
    fn colors(&self) -> [Option<ColorState>; 1] {
        let color = ColorState::new(self.format.format);
        [Some(match self.blended {
            true => color.blended(OVER),
            false => color,
        })]
    }

    fn pipeline<'a>(
        &self,
        fragment: &'static str,
        colors: &'a [Option<ColorState>],
    ) -> RenderPipelineSpec<'a> {
        RenderPipelineSpec::new(VERTEX, fragment, colors).multisampled(self.samples)
    }

    fn target(&self) -> TargetSpec {
        match self.samples {
            1 => TargetSpec::read(self.format.format, [SIDE, SIDE]),
            n => TargetSpec::plain(self.format.format, [SIDE, SIDE]).multisampled(n),
        }
    }

    fn resolve(&self) -> TargetSpec {
        TargetSpec::read(self.format.format, [SIDE, SIDE])
    }
}

/// The steps of one draw: `pipeline`, the triangle's corners at vertex buffer 0, the stops at
/// fragment buffer 0 and the info at 1, and the triangle.
fn gradient_draw(
    pipeline: Handle,
    corners: &[[f32; 2]; 3],
    (stops, info): (Handle, Handle),
) -> [RenderStep; 5] {
    [
        RenderStep::Pipeline(pipeline),
        RenderStep::VertexBytes {
            bytes: corner_bytes(corners),
            index: 0,
        },
        RenderStep::FragmentBuffer {
            buffer: stops,
            offset: 0,
            index: 0,
        },
        RenderStep::FragmentBuffer {
            buffer: info,
            offset: 0,
            index: 1,
        },
        RenderStep::triangle(),
    ]
}

impl Case for GradientDraw {
    fn name(&self) -> &'static str {
        self.name
    }

    fn metal_source(&self) -> MetalSource {
        MetalSource::Run
    }

    /// The notices a draw into such a target may announce, none of which loosens a code (every
    /// texel still compares exactly): `shared-in-system-memory` (the read-back buffer is `Shared`;
    /// a device without unified memory announces it), the kept encoding of a blit through a
    /// `BGRA10_XR` texture Metal keeps compressed (a draw stores canonical codes, so the blit
    /// moves the same bytes either way), `half-output-fast-math-blend` (the fixture is a
    /// fast-math module whose `half4` gradient a pipeline blends), `extended-range-bits-kept` (a
    /// draw in a pass over such a color; every texel is written by a clear or a draw first), and
    /// `extended-range-blends-stored-codes` (the second draw blends over the first's color where
    /// the pass cannot hold its `half` shadow). At four samples, what the pass's shape announces:
    /// `shader-resolve` and `extended-range-resolve` (the runtime's own resolve averages the
    /// codes), `dont-care-store` (the multisampled attachment of `MultisampleResolve`, on a driver
    /// measured to keep one of Metal's answers), and on a device without multisampled storage
    /// images `sample-planes` and `fetch-through-scratch` (the interlocked blend reads and writes
    /// the samples as planes, through a scratch image the target's usage needs). A blended row also
    /// declares `render-pass-split` (`crate::replay::interlocked_split_notice`): its second draw
    /// blends in the pixel interlock over what the first wrote there.
    fn allowances(&self) -> Vec<Allowance> {
        let mut named = vec![crate::replay::shared_memory_notice()];
        named.extend(super::compressed_blits::kept_encoding_notice(
            self.format.format.as_raw(),
        ));
        let mut kinds = vec![
            Degradation::HalfOutputFastMathBlend,
            Degradation::ExtendedRangeBitsKept,
        ];
        if self.blended {
            kinds.push(Degradation::ExtendedRangeBlendsStoredCodes);
        }
        if self.samples > 1 {
            kinds.extend([
                Degradation::ShaderResolve,
                Degradation::ExtendedRangeResolve,
                Degradation::DontCareStore,
                Degradation::SamplePlanes,
                Degradation::FetchThroughScratch,
            ]);
        }
        named.extend(
            kinds
                .into_iter()
                .map(|kind| crate::replay::notice_allowance(kind.code())),
        );
        if self.blended {
            named.push(crate::replay::interlocked_split_notice());
        }
        named
    }

    fn shaders(&self) -> Result<Vec<ShaderRef>, String> {
        shader_refs(GRADIENTS, &[VERTEX, RADIAL_FLOAT, LINEAR_HALF])
    }

    fn inputs(&self) -> Result<Vec<Input>, String> {
        let mut inputs = fixture_inputs("gradients", GRADIENTS)?;
        let colors = self.colors();
        inputs.extend([
            Input::bytes("format", &self.format.format.as_raw().to_le_bytes()),
            Input::bytes("blend", &OVER.words()),
            Input::bytes(
                "pipeline-radial",
                &self.pipeline(RADIAL_FLOAT, &colors).words(),
            ),
            Input::bytes(
                "pipeline-linear",
                &self.pipeline(LINEAR_HALF, &colors).words(),
            ),
            Input::bytes("target", &self.target().words()),
            Input::bytes(
                "clear",
                &CLEAR
                    .iter()
                    .flat_map(|c| c.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
            Input::bytes("corners-full", &corner_bytes(&FULL)),
            Input::bytes("corners-part", &corner_bytes(&PART)),
            Input::bytes("buffer:radial-stops", &RADIAL.stop_bytes()),
            Input::bytes("buffer:radial-info", &RADIAL.info_bytes()),
            Input::bytes("buffer:linear-stops", &LINEAR.stop_bytes()),
            Input::bytes("buffer:linear-info", &LINEAR.info_bytes()),
        ]);
        if self.samples > 1 {
            inputs.push(Input::bytes("resolve", &self.resolve().words()));
        }
        Ok(inputs)
    }

    fn run(&self, side: &mut dyn Side, sites: &mut Sites) -> Result<(), Abort> {
        let gradients = fixture(GRADIENTS, VERTEX)
            .loaded()
            .map_err(|why| not_read("library/gradients", why))?;
        let library = made(
            sites,
            "library/gradients",
            side.library(&gradients.source()),
        )?;
        let colors = self.colors();
        let radial = made(
            sites,
            "pipeline/radial",
            side.render_pipeline(library, &self.pipeline(RADIAL_FLOAT, &colors)),
        )?;
        let linear = made(
            sites,
            "pipeline/linear",
            side.render_pipeline(library, &self.pipeline(LINEAR_HALF, &colors)),
        )?;
        let radial_stops = made(
            sites,
            "buffer/radial-stops",
            side.buffer(&RADIAL.stop_bytes()),
        )?;
        let radial_info = made(
            sites,
            "buffer/radial-info",
            side.buffer(&RADIAL.info_bytes()),
        )?;
        let linear_stops = made(
            sites,
            "buffer/linear-stops",
            side.buffer(&LINEAR.stop_bytes()),
        )?;
        let linear_info = made(
            sites,
            "buffer/linear-info",
            side.buffer(&LINEAR.info_bytes()),
        )?;
        let target = made(sites, "target", side.target(&self.target()))?;
        let mut steps = Vec::new();
        steps.extend(gradient_draw(radial, &FULL, (radial_stops, radial_info)));
        steps.extend(gradient_draw(linear, &PART, (linear_stops, linear_info)));
        let (color, read) = match self.samples {
            1 => (PassColor::cleared(target, CLEAR), target),
            _ => {
                let resolve = made(sites, "resolve", side.target(&self.resolve()))?;
                let color = PassColor::cleared(target, CLEAR)
                    .resolving(resolve)
                    .stored(StoreAction::MultisampleResolve);
                (color, resolve)
            }
        };
        let pass = RenderPass::clearing(vec![color]).drawing(steps);
        let mut work = vec![RenderWork::Pass(pass)];
        let bytes = (SIDE * SIDE * self.format.texel) as usize;
        let out = made(sites, "buffer/texels", side.zeroed_buffer(bytes))?;
        work.push(RenderWork::ReadTexture {
            texture: read,
            into: out,
            offset: 0,
            row_pitch: SIDE * self.format.texel,
            size: [SIDE, SIDE],
            plane: Plane::Whole,
            at: ReadAt::BASE,
        });
        let outcome = call(sites, "render", side.render(&work), |_| Value::Absent)?;
        program(sites, "program", &outcome, &work);
        let texels = call(sites, "read-back", side.read_buffer(out, 0, bytes), |_| {
            Value::Absent
        })?;
        sites.push(Node::leaf(
            "texels",
            Observed::bytes(texels, texel_layout(&self.format, [SIDE, SIDE])),
        ));
        Ok(())
    }
}
