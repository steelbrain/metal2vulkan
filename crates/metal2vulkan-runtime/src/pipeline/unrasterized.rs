//! Metal's refusals of a render pipeline state by what its vertex function returns: one with
//! rasterization disabled whose vertex function returns a value, and one that rasterizes whose
//! vertex function returns none, which Metal makes in OpenGL mode.
//!
//! - **What Metal refuses** (on an M4 under macOS 27.0.1). `newRenderPipelineStateWithDescriptor:`
//!   with `rasterizationEnabled` NO answers nil with an `NSError` where the vertex function's
//!   return type is not `void`: a struct of varyings, a struct of its position alone, or a bare
//!   position, with or without `supportIndirectCommandBuffers`. A `void` vertex function is made.
//!   With `supportIndirectCommandBuffers` this refusal comes first: a vertex function that returns
//!   a position read from a texture gets this error, not the indirect command buffers' one.
//! - **The answer.** [`Error::MetalError`] with [`Problem::UnrasterizedVertexReturn`] and Metal's
//!   error: code 3, its description, `userInfo` holding only `NSLocalizedDescription`, and the
//!   domain of Metal's GPU driver class, as `indirect_arguments.rs` answers it.
//! - **Tessellation too** (on an M4 under macOS 27.0.1). A post-tessellation
//!   vertex function that returns a value is refused alike, with or without a fragment function, a
//!   color attachment or `supportIndirectCommandBuffers`, and one that returns nothing is made,
//!   whatever the descriptor's tessellation state.
//! - **Where it binds.** A state that rasterizes nothing covers no sample, so it stores into no
//!   attachment: it binds in a pass of any colors it does not name and of any sample count, and
//!   its draws leave every attachment as it was. Metal's API validation accepts it, and the
//!   attachments' stored bytes and the visibility counts are those of the pass without the draw
//!   (measured on an M3 Max under macOS 27.0.1, over passes of 1 and 8 colors of mixed formats, 2
//!   and 4 samples, depth, stencil or both, a two-slice target, vertex amplification, two
//!   viewports, a memoryless attachment, imageblocks and a tile dispatch; empty draws alone).
//!   `ShapeInputs::check_pipeline` (`render.rs`) therefore refuses a state of neither colors nor
//!   rasterization in neither case, where a state that rasterizes without a fragment function is
//!   still refused a color it lacks, and a mesh render pipeline's state is refused as before (both
//!   unmeasured). A non-empty draw of such a state is unmeasured on Metal and draws nothing here.
//! - **A rasterizing `void` vertex function** (on an M3 Max under macOS 27.0.1, probe
//!   `conformance/probes/opengl_mode.m`). With `rasterizationEnabled` YES, a vertex function whose
//!   return type is `void` is refused, with or without a fragment function: code 3, "RasterizationEnabled
//!   is true but the vertex shader's return type is void" ([`Problem::RasterizedVoidVertex`]).
//!   With `openGLModeEnabled` (`PrivateRenderState::open_gl_mode_enabled`) the state is made, and
//!   its draws run the vertex function (its stores land) and rasterize nothing (every color stays
//!   as the pass left it), with or without a fragment function: the runtime makes it as the state
//!   of its descriptor with rasterization disabled, its fragment function dropped, which binds and
//!   draws as above. A post-tessellation vertex function is taken alike, unmeasured.
//! - **Decided from the AIR.** Whether the vertex function returns a value is the compile layer's
//!   answer (`RenderPipelineDescriptor::vertex_returns_value`); one it refuses (a stage it already
//!   compiled, which a render descriptor's Metal form never holds) decides nothing.

use super::indirect_arguments::{CODE, DOMAIN};
use crate::error::{Error, MetalError, Problem};
use metal2vulkan::error::Error as CompileError;
use metal2vulkan::pipeline::RenderPipelineDescriptor;

/// Metal's `localizedDescription` for a state with rasterization disabled whose vertex function
/// returns a value.
const DESCRIPTION: &str =
    "RasterizationEnabled is false but the vertex shader's return type is not void";

/// Metal's `localizedDescription` for a rasterizing state whose vertex function returns none.
const VOID: &str = "RasterizationEnabled is true but the vertex shader's return type is void";

/// Whether the state of `descriptor` rasterizes, its vertex function returning a value or not as
/// `returns` answers under the creation's budget: not with rasterization disabled, nor in OpenGL
/// mode where its vertex function returns none (the module docs). Refused in `call` where Metal
/// refuses the state; a refused answer decides nothing, the state rasterizing as `descriptor`
/// says.
pub(crate) fn rasterizes(
    call: &'static str,
    descriptor: &RenderPipelineDescriptor,
    returns: Result<bool, CompileError>,
) -> Result<bool, Error> {
    let refused = |problem, description| {
        Err(Error::MetalError(MetalError::new(
            call,
            problem,
            DOMAIN,
            CODE,
            description,
        )))
    };
    match (descriptor.rasterization_enabled, returns) {
        (false, Ok(true)) => refused(Problem::UnrasterizedVertexReturn, DESCRIPTION),
        (true, Ok(false)) if !descriptor.private.open_gl_mode_enabled => {
            refused(Problem::RasterizedVoidVertex, VOID)
        }
        (true, Ok(false)) => Ok(false),
        (rasterizes, _) => Ok(rasterizes),
    }
}
