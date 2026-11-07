//! Metal's refusals of a render pipeline state for what its fragment function does once linked,
//! with or without rasterization.
//!
//! - **Metal links the fragment function whether or not the pipeline rasterizes** (on an M4 under
//!   macOS 27.0.1): with rasterization disabled, a state whose fragment function reads a stage
//!   input is refused where the read reaches a color attachment 0 writes, and made without one; a
//!   constant-color fragment function is made. The runtime takes the fragment function there too:
//!   these refusals and the indirect command buffers' see it, and the compile drops it (nothing
//!   rasterizes).
//! - **Depth without a depth attachment.** A fragment function with a `[[depth(...)]]` output in
//!   a pipeline without a depth attachment is refused, with or without rasterization, a color
//!   attachment or `supportIndirectCommandBuffers`, and before the indirect command buffers'
//!   refusal: code 3, "depthAttachmentPixelFormat is not valid and shader writes to depth"
//!   ([`Problem::DepthWrittenWithoutAttachment`]). With `openGLModeEnabled`
//!   (`PrivateRenderState::open_gl_mode_enabled`) the state is made, and draws its colors as
//!   without the output (on an M3 Max under macOS 27.0.1, probe
//!   `conformance/probes/opengl_mode.m`): the runtime makes it so, Vulkan dropping the depth of a
//!   pass without a depth attachment.
//! - **Stage inputs the vertex function does not write.** A fragment function's stage inputs its
//!   linking keeps (`RenderPipelineDescriptor::unwritten_fragment_inputs`: those a kept instruction
//!   reads, the colors its attachments write kept) whose user key no vertex output writes are
//!   refused, after the indirect command buffers' refusal: code 3, "Fragment input(s) `c,d`
//!   mismatching vertex shader output type(s) or not written by vertex shader", naming the inputs
//!   by their MSL member names in declaration order, joined by commas
//!   ([`Problem::UnwrittenFragmentInputs`]). An input the linking drops is not named, and one that
//!   is the only reason refuses nothing (rasterization on, a vertex function writing `c` and a
//!   fragment function reading `c` and `d` into color 0: `d` alone; without color 0, made). Where
//!   only a rule kept beyond Metal keeps such an input, the answer is [`Error::Unproven`]
//!   ([`Problem::UnwrittenFragmentInputsUnproven`]).
//! - **Tessellation and mesh pipelines alike** (on an M4 under macOS 27.0.1): a
//!   post-tessellation vertex function's unwritten input is refused with the same text; a mesh
//!   function's with "mesh shader" in place of "vertex shader" (per-vertex and per-primitive data
//!   both write a key). An input the linking drops is made in all three: the compile reads it as
//!   zero.
//! - **Not measured:** a mesh pipeline's depth without a depth attachment (not refused), the
//!   order of a mesh pipeline's refusals (taken as a render pipeline's), and the order of these
//!   refusals beside a vertex function that returns a value with rasterization disabled (that
//!   one comes first).

use super::indirect_arguments::{CODE, DOMAIN};
use crate::error::{Error, MetalError, Problem, Unproven};
use metal2vulkan::error::Error as CompileError;
use metal2vulkan::reflection::UnwrittenFragmentInputs;

/// Metal's `localizedDescription` for a fragment function writing depth without a depth
/// attachment.
const DEPTH: &str = "depthAttachmentPixelFormat is not valid and shader writes to depth";

/// Refused in `call` (the module docs): a fragment function that `writes` depth without a depth
/// attachment, as the descriptor answers under the creation's budget; a refused answer decides
/// nothing.
pub(crate) fn check_depth(
    call: &'static str,
    writes: Result<bool, CompileError>,
) -> Result<(), Error> {
    match writes {
        Ok(true) => Err(Error::MetalError(MetalError::new(
            call,
            Problem::DepthWrittenWithoutAttachment,
            DOMAIN,
            CODE,
            DEPTH,
        ))),
        Ok(false) | Err(_) => Ok(()),
    }
}

/// Refused in `call` (the module docs): the fragment function's stage inputs the linking keeps
/// and the `shader` function (`"vertex"` or `"mesh"`, as Metal's text names it) does not write
/// (`unwritten`, the descriptor's answer under the creation's budget); a refused answer decides
/// nothing.
pub(crate) fn check_inputs(
    call: &'static str,
    shader: &'static str,
    unwritten: Result<Option<UnwrittenFragmentInputs>, CompileError>,
) -> Result<(), Error> {
    let Ok(Some(unwritten)) = unwritten else {
        return Ok(());
    };
    let inputs = unwritten.names.into_boxed_slice();
    Err(match unwritten.gap {
        None => {
            let description = format!(
                "Fragment input(s) `{}` mismatching {shader} shader output type(s) or not written \
                 by {shader} shader",
                inputs.join(",")
            );
            Error::MetalError(MetalError::with_description(
                call,
                Problem::UnwrittenFragmentInputs { inputs },
                DOMAIN,
                CODE,
                description,
            ))
        }
        Some(reason) => Error::Unproven(Unproven::new(
            call,
            Problem::UnwrittenFragmentInputsUnproven { inputs, reason },
        )),
    })
}
